use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use casc_lib::blte::decoder::decode_blte_with_keys;
use casc_lib::extract::{CascStorage, OpenConfig};
use casc_lib::listfile::parser::Listfile;
use casc_lib::root::flags::LocaleFlags;

pub trait AssetSource {
    fn read(&self, wow_path: &str) -> Result<Vec<u8>>;
    fn contains(&self, wow_path: &str) -> bool;
    fn path_for_fdid(&self, fdid: u32) -> Option<String>;
    fn fdid_for_path(&self, wow_path: &str) -> Option<u32>;
    fn read_db2_csv(&self, table: &str) -> Result<Vec<u8>>;
    fn build(&self) -> String;
    fn product(&self) -> String;
}

pub enum ClientSource {
    Casc(Box<CascClient>),
    Loose(LooseClient),
}

impl ClientSource {
    pub fn open(
        client: &Path,
        product: &str,
        locale: &str,
        listfile: Option<&Path>,
        cache: &Path,
    ) -> Result<Self> {
        if client.join(".build.info").is_file() {
            return Ok(Self::Casc(Box::new(CascClient::open(
                client, product, locale, listfile, cache,
            )?)));
        }
        if client.is_dir() {
            return Ok(Self::Loose(LooseClient::open(client)?));
        }
        bail!("client path does not exist: {}", client.display());
    }
}

impl AssetSource for ClientSource {
    fn read(&self, wow_path: &str) -> Result<Vec<u8>> {
        match self {
            Self::Casc(source) => source.read(wow_path),
            Self::Loose(source) => source.read(wow_path),
        }
    }

    fn contains(&self, wow_path: &str) -> bool {
        match self {
            Self::Casc(source) => source.contains(wow_path),
            Self::Loose(source) => source.contains(wow_path),
        }
    }

    fn path_for_fdid(&self, fdid: u32) -> Option<String> {
        match self {
            Self::Casc(source) => source.path_for_fdid(fdid),
            Self::Loose(source) => source.path_for_fdid(fdid),
        }
    }

    fn fdid_for_path(&self, wow_path: &str) -> Option<u32> {
        match self {
            Self::Casc(source) => source.fdid_for_path(wow_path),
            Self::Loose(source) => source.fdid_for_path(wow_path),
        }
    }

    fn read_db2_csv(&self, table: &str) -> Result<Vec<u8>> {
        match self {
            Self::Casc(source) => source.read_db2_csv(table),
            Self::Loose(source) => source.read_db2_csv(table),
        }
    }

    fn build(&self) -> String {
        match self {
            Self::Casc(source) => source.build(),
            Self::Loose(source) => source.build(),
        }
    }

    fn product(&self) -> String {
        match self {
            Self::Casc(source) => source.product(),
            Self::Loose(source) => source.product(),
        }
    }
}

pub struct CascClient {
    storage: CascStorage,
    locale: LocaleFlags,
    cache: PathBuf,
    http: reqwest::blocking::Client,
}

impl CascClient {
    fn open(
        client: &Path,
        product: &str,
        locale: &str,
        listfile: Option<&Path>,
        cache: &Path,
    ) -> Result<Self> {
        let locale = parse_locale(locale)?;
        let config = OpenConfig {
            install_dir: client.to_path_buf(),
            product: Some(product.to_owned()),
            keyfile: None,
            listfile: listfile.map(Path::to_path_buf),
            output_dir: Some(cache.to_path_buf()),
        };
        let storage = CascStorage::open(&config).with_context(|| {
            format!(
                "failed to open CASC client {} with product {}",
                client.display(),
                product
            )
        })?;
        if storage.listfile.is_empty() {
            bail!("CASC listfile is empty; pass --listfile with a community-listfile CSV");
        }
        fs::create_dir_all(cache)?;
        fs::write(cache.join("build.txt"), &storage.info().version)?;
        fs::write(cache.join("product.txt"), &storage.info().product)?;
        let cached_listfile = cache.join("listfile.csv");
        if !cached_listfile.is_file() {
            let source = listfile
                .map(Path::to_path_buf)
                .unwrap_or_else(|| cache.join(".casc-meta/listfile.csv"));
            fs::copy(&source, &cached_listfile)
                .with_context(|| format!("failed to cache listfile from {}", source.display()))?;
        }
        Ok(Self {
            storage,
            locale,
            cache: cache.to_path_buf(),
            http: reqwest::blocking::Client::new(),
        })
    }

    fn read_from_cdn(&self, ekey: &[u8; 16]) -> Result<Vec<u8>> {
        let key = hex(ekey);
        let cache_path = self.cache.join("decoded").join(format!("{key}.bin"));
        if cache_path.is_file() {
            return fs::read(&cache_path)
                .with_context(|| format!("failed to read CDN cache {}", cache_path.display()));
        }
        let mut failures = Vec::new();
        for host in &self.storage.build_info.cdn_hosts {
            let url = format!(
                "https://{}/{}/data/{}/{}/{}",
                host,
                self.storage.build_info.cdn_path,
                &key[..2],
                &key[2..4],
                key
            );
            let response = match self.http.get(&url).send() {
                Ok(response) => response,
                Err(error) => {
                    failures.push(format!("{host}: {error}"));
                    continue;
                }
            };
            if !response.status().is_success() {
                failures.push(format!("{host}: HTTP {}", response.status()));
                continue;
            }
            let bytes = response
                .bytes()
                .with_context(|| format!("failed to download CDN object {key}"))?;
            let data = decode_casc_payload(&bytes, &self.storage)?;
            if let Some(parent) = cache_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&cache_path, &data)?;
            return Ok(data);
        }
        bail!("CDN object {key} unavailable: {}", failures.join("; "))
    }

    fn read_from_wago(&self, fdid: u32) -> Result<Vec<u8>> {
        let info = self.storage.info();
        let cache_path = self.wago_cache_path(fdid, &info.version);
        if cache_path.is_file() {
            return fs::read(&cache_path)
                .with_context(|| format!("failed to read Wago cache {}", cache_path.display()));
        }
        let response = self
            .http
            .get(format!(
                "https://wago.tools/api/casc/{fdid}?version={}&product={}",
                info.version, info.product
            ))
            .send()
            .with_context(|| format!("failed to request FDID {fdid} from Wago"))?
            .error_for_status()
            .with_context(|| format!("Wago has no FDID {fdid} for {}", info.version))?;
        let data = response
            .bytes()
            .with_context(|| format!("failed to download FDID {fdid} from Wago"))?
            .to_vec();
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, &data)?;
        Ok(data)
    }

    fn wago_cache_path(&self, fdid: u32, version: &str) -> PathBuf {
        self.cache
            .join("decoded")
            .join(format!("wago-{}-{fdid}.bin", version.replace('.', "_")))
    }

    fn asset_cache_path(&self, wow_path: &str) -> PathBuf {
        wow_path
            .split('/')
            .fold(self.cache.join("files"), |path, component| {
                path.join(component)
            })
    }

    fn cache_asset(&self, wow_path: &str, data: Vec<u8>) -> Result<Vec<u8>> {
        let path = self.asset_cache_path(wow_path);
        if !path.is_file() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, &data)
                .with_context(|| format!("failed to cache asset {}", path.display()))?;
        }
        Ok(data)
    }

    fn read_db2_csv(&self, table: &str) -> Result<Vec<u8>> {
        let version = self.storage.info().version;
        let cache_path = self
            .cache
            .join("db2")
            .join(version.replace('.', "_"))
            .join(format!("{table}.csv"));
        if cache_path.is_file() {
            return fs::read(&cache_path)
                .with_context(|| format!("failed to read DB2 cache {}", cache_path.display()));
        }
        let response = self
            .http
            .get(format!(
                "https://wago.tools/db2/{table}/csv?build={version}"
            ))
            .header(reqwest::header::USER_AGENT, "wow2strelka/0.1")
            .send()
            .with_context(|| format!("failed to request {table} metadata from Wago"))?
            .error_for_status()
            .with_context(|| format!("Wago has no {table} metadata for {version}"))?;
        let data = response.bytes()?.to_vec();
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, &data)?;
        Ok(data)
    }
}

impl AssetSource for CascClient {
    fn read(&self, wow_path: &str) -> Result<Vec<u8>> {
        let normalized = normalize(wow_path);
        let cached = self.asset_cache_path(&normalized);
        if cached.is_file() {
            return fs::read(&cached)
                .with_context(|| format!("failed to read asset cache {}", cached.display()));
        }
        let fdid = self
            .storage
            .listfile
            .fdid(&normalized)
            .with_context(|| format!("asset is absent from listfile: {normalized}"))?;
        let wago_cache = self.wago_cache_path(fdid, &self.storage.info().version);
        if wago_cache.is_file() {
            let data = fs::read(&wago_cache)
                .with_context(|| format!("failed to read Wago cache {}", wago_cache.display()))?;
            return self.cache_asset(&normalized, data);
        }
        let root = self
            .storage
            .root
            .find_by_fdid(fdid, self.locale)
            .with_context(|| format!("asset is absent from CASC root: {normalized}"))?;
        let encoding = self
            .storage
            .encoding
            .find_ekey(&root.ckey)
            .with_context(|| format!("asset has no CASC encoding entry: {normalized}"))?;
        let mut failures = Vec::new();
        for ekey in &encoding.ekeys {
            if let Some(index) = self.storage.index.find(ekey) {
                if index.size > 256 * 1024 * 1024 {
                    failures.push(format!("implausible CASC asset size {}", index.size));
                    continue;
                }
                match self.storage.data.read_raw(
                    index.archive_number,
                    index.archive_offset,
                    index.size,
                ) {
                    Ok(raw) => match decode_casc_payload(raw, &self.storage) {
                        Ok(data) => return self.cache_asset(&normalized, data),
                        Err(error) => failures.push(error.to_string()),
                    },
                    Err(error) => failures.push(error.to_string()),
                }
            }
            match self.read_from_cdn(ekey) {
                Ok(data) => return self.cache_asset(&normalized, data),
                Err(error) => failures.push(error.to_string()),
            }
        }
        match self.read_from_wago(fdid) {
            Ok(data) => return self.cache_asset(&normalized, data),
            Err(error) => failures.push(error.to_string()),
        }
        bail!(
            "failed to extract {normalized} (FDID {fdid}) using {} encoding key(s): {}",
            encoding.ekeys.len(),
            failures.join("; ")
        )
    }

    fn contains(&self, wow_path: &str) -> bool {
        self.storage.listfile.fdid(&normalize(wow_path)).is_some()
    }

    fn path_for_fdid(&self, fdid: u32) -> Option<String> {
        self.storage.listfile.path(fdid).map(normalize)
    }

    fn fdid_for_path(&self, wow_path: &str) -> Option<u32> {
        self.storage.listfile.fdid(&normalize(wow_path))
    }

    fn read_db2_csv(&self, table: &str) -> Result<Vec<u8>> {
        CascClient::read_db2_csv(self, table)
    }

    fn build(&self) -> String {
        self.storage.info().version
    }

    fn product(&self) -> String {
        self.storage.info().product
    }
}

fn decode_casc_payload(raw: &[u8], storage: &CascStorage) -> Result<Vec<u8>> {
    let payload = if raw.starts_with(b"BLTE") {
        raw
    } else if raw.len() >= 34 && raw[30..].starts_with(b"BLTE") {
        &raw[30..]
    } else {
        bail!("archive entry has no BLTE header");
    };
    decode_blte_with_keys(payload, Some(&storage.keystore)).map_err(Into::into)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    result
}

pub struct LooseClient {
    root: PathBuf,
    metadata_root: PathBuf,
    listfile: Option<Listfile>,
    build: String,
    product: String,
}

impl LooseClient {
    fn open(root: &Path) -> Result<Self> {
        let files = root.join("files");
        let listfile = [
            root.join("listfile.csv"),
            root.join(".casc-meta/listfile.csv"),
        ]
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| Listfile::load(&path))
        .transpose()?;
        Ok(Self {
            root: if files.is_dir() {
                files
            } else {
                root.to_path_buf()
            },
            metadata_root: root.to_path_buf(),
            listfile,
            build: read_label(root.join("build.txt"), "loose-files")?,
            product: read_label(root.join("product.txt"), "extracted")?,
        })
    }

    fn path(&self, wow_path: &str) -> PathBuf {
        normalize(wow_path)
            .split('/')
            .fold(self.root.clone(), |path, component| path.join(component))
    }

    fn decoded_path(&self, wow_path: &str) -> Option<PathBuf> {
        let fdid = self.fdid_for_path(wow_path)?;
        Some(
            self.metadata_root
                .join("decoded")
                .join(format!("wago-{}-{fdid}.bin", self.build.replace('.', "_"))),
        )
    }
}

impl AssetSource for LooseClient {
    fn read(&self, wow_path: &str) -> Result<Vec<u8>> {
        let path = self.path(wow_path);
        if path.is_file() {
            return fs::read(&path).with_context(|| format!("failed to read {}", path.display()));
        }
        let decoded = self.decoded_path(wow_path);
        let path = decoded.as_deref().unwrap_or(&path);
        fs::read(path).with_context(|| format!("failed to read {}", path.display()))
    }

    fn contains(&self, wow_path: &str) -> bool {
        self.path(wow_path).is_file()
            || self
                .decoded_path(wow_path)
                .is_some_and(|path| path.is_file())
    }

    fn path_for_fdid(&self, fdid: u32) -> Option<String> {
        self.listfile.as_ref()?.path(fdid).map(normalize)
    }

    fn fdid_for_path(&self, wow_path: &str) -> Option<u32> {
        self.listfile.as_ref()?.fdid(&normalize(wow_path))
    }

    fn read_db2_csv(&self, table: &str) -> Result<Vec<u8>> {
        let path = self
            .metadata_root
            .join("db2")
            .join(self.build.replace('.', "_"))
            .join(format!("{table}.csv"));
        fs::read(&path).with_context(|| format!("failed to read DB2 cache {}", path.display()))
    }

    fn build(&self) -> String {
        self.build.clone()
    }

    fn product(&self) -> String {
        self.product.clone()
    }
}

fn read_label(path: PathBuf, fallback: &str) -> Result<String> {
    if !path.is_file() {
        return Ok(fallback.to_owned());
    }
    Ok(fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?
        .trim()
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_client_works_without_casc() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("files/world/maps")).unwrap();
        fs::create_dir_all(temp.path().join("db2/5_5_4_69585")).unwrap();
        fs::create_dir_all(temp.path().join("decoded")).unwrap();
        fs::write(temp.path().join("files/world/maps/test.adt"), b"adt").unwrap();
        fs::write(
            temp.path().join("listfile.csv"),
            "42;World/Maps/Test.adt\n43;Creature/Test.m2\n",
        )
        .unwrap();
        fs::write(temp.path().join("decoded/wago-5_5_4_69585-43.bin"), b"m2").unwrap();
        fs::write(temp.path().join("build.txt"), "5.5.4.69585").unwrap();
        fs::write(temp.path().join("product.txt"), "wow_classic").unwrap();
        fs::write(temp.path().join("db2/5_5_4_69585/Test.csv"), b"ID\n1\n").unwrap();

        let client = LooseClient::open(temp.path()).unwrap();
        assert_eq!(client.read("WORLD\\MAPS\\TEST.ADT").unwrap(), b"adt");
        assert_eq!(client.read("creature/test.m2").unwrap(), b"m2");
        assert_eq!(
            client.path_for_fdid(42).as_deref(),
            Some("world/maps/test.adt")
        );
        assert_eq!(client.fdid_for_path("world/maps/test.adt"), Some(42));
        assert_eq!(client.read_db2_csv("Test").unwrap(), b"ID\n1\n");
        assert_eq!(client.build(), "5.5.4.69585");
        assert_eq!(client.product(), "wow_classic");
    }
}

pub fn normalize(path: &str) -> String {
    path.replace('\\', "/")
        .trim_start_matches('/')
        .to_lowercase()
}

fn parse_locale(locale: &str) -> Result<LocaleFlags> {
    let value = match locale.to_ascii_lowercase().as_str() {
        "enus" => LocaleFlags::EN_US,
        "engb" => LocaleFlags::EN_GB,
        "ruru" => LocaleFlags::RU_RU,
        "dede" => LocaleFlags::DE_DE,
        "frfr" => LocaleFlags::FR_FR,
        "eses" => LocaleFlags::ES_ES,
        "esmx" => LocaleFlags::ES_MX,
        "ptbr" => LocaleFlags::PT_BR,
        "itit" => LocaleFlags::IT_IT,
        "kokr" => LocaleFlags::KO_KR,
        "zhcn" => LocaleFlags::ZH_CN,
        "zhtw" => LocaleFlags::ZH_TW,
        "all" => LocaleFlags::ALL,
        _ => bail!("unsupported locale '{locale}'"),
    };
    Ok(value)
}
