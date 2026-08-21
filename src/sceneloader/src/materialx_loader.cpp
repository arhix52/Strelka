#include <strelka/sceneloader/materialx_loader.h>

#include <log.h>
#include <paths.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Look.h>
#include <MaterialXCore/Material.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

#include <algorithm>
#include <ranges>
#include <cmath>
#include <unordered_map>

namespace mx = MaterialX;

namespace oka::mtlx
{
namespace
{

/// What an input resolved to.
struct Resolved
{
    bool isConstant = false;
    mx::ValuePtr value;
    std::string texture; // absolute path, empty when none
    std::string unsupportedCategory;
};

/// Follows an input to a constant, to a single image, or to nothing this loader
/// can express.
///
/// The normalmap step is why this is a walk rather than a lookup: a normal input
/// is conventionally <normalmap in="<image>">, and the image is one hop further
/// than every other slot.
Resolved resolveInput(const mx::InputPtr& input, const mx::FilePath& docDir)
{
    Resolved out;
    if (!input)
    {
        return out;
    }

    if (input->hasValue())
    {
        out.isConstant = true;
        out.value = input->getValue();
        return out;
    }

    mx::NodePtr node;
    if (const mx::OutputPtr connected = input->getConnectedOutput())
    {
        node = connected->getConnectedNode();
    }
    if (!node)
    {
        node = input->getConnectedNode();
    }
    if (!node)
    {
        return out;
    }

    if (node->getCategory() == "normalmap")
    {
        node = node->getConnectedNode("in");
        if (!node)
        {
            out.unsupportedCategory = "normalmap(unconnected)";
            return out;
        }
    }

    const std::string category = node->getCategory();
    if (category == "image" || category == "tiledimage")
    {
        const mx::InputPtr file = node->getInput("file");
        if (file)
        {
            const std::string name = file->getValueString();
            if (!name.empty())
            {
                // MaterialX resolves a filename input against the document it
                // came from, so a path is only meaningful next to its .mtlx.
                // Made absolute here because the renderer joins texture paths
                // with the *scene's* search path, and the two directories are
                // not the same one for a shared material library.
                out.texture = (docDir / mx::FilePath(name)).asString();
            }
        }
        return out;
    }

    out.unsupportedCategory = category;
    return out;
}

float asFloat(const mx::ValuePtr& v, float fallback)
{
    if (!v)
        return fallback;
    if (v->isA<float>())
        return v->asA<float>();
    if (v->isA<int>())
        return static_cast<float>(v->asA<int>());
    if (v->isA<bool>())
        return v->asA<bool>() ? 1.0f : 0.0f;
    if (v->isA<mx::Color3>())
        return v->asA<mx::Color3>()[0];
    return fallback;
}

OpenPBRColor asColor(const mx::ValuePtr& v, OpenPBRColor fallback)
{
    if (!v)
        return fallback;
    if (v->isA<mx::Color3>())
    {
        const mx::Color3 c = v->asA<mx::Color3>();
        return OpenPBRColor{ c[0], c[1], c[2] };
    }
    if (v->isA<mx::Vector3>())
    {
        const mx::Vector3 c = v->asA<mx::Vector3>();
        return OpenPBRColor{ c[0], c[1], c[2] };
    }
    if (v->isA<float>())
    {
        const float f = v->asA<float>();
        return OpenPBRColor{ f, f, f };
    }
    return fallback;
}

bool asBool(const mx::ValuePtr& v, bool fallback)
{
    if (v && v->isA<bool>())
        return v->asA<bool>();
    return fallback;
}

/// Where a shader input's value goes, and where its texture goes.
///
/// Pointers to members rather than byte offsets: the table is data, and the
/// alternative -- offsetof plus a reinterpret_cast at every write -- puts the
/// field's type in one place and its address in another, so a mistyped entry
/// compiles and corrupts the neighbouring parameter instead of failing.
///
/// One table per shading model rather than one merged table, because the two
/// disagree about names in both directions (standard_surface's `sheen` is
/// OpenPBR's `fuzz`, its `specular_IOR` is `specular_ior`) and a merged table
/// would silently accept a name from the wrong model.
struct Binding
{
    float OpenPBRParams::* asFloatMember = nullptr;
    /// The sine half of a rotation; set only for RotationTurns bindings.
    float OpenPBRParams::* asSinMember = nullptr;
    OpenPBRColor OpenPBRParams::* asColorMember = nullptr;
    unsigned int OpenPBRParams::* asBoolMember = nullptr;
    int textureSlot = -1;
};

Binding bindFloat(float OpenPBRParams::* member, int slot = -1)
{
    Binding b;
    b.asFloatMember = member;
    b.textureSlot = slot;
    return b;
}

Binding bindColor(OpenPBRColor OpenPBRParams::* member, int slot = -1)
{
    Binding b;
    b.asColorMember = member;
    b.textureSlot = slot;
    return b;
}

Binding bindBool(unsigned int OpenPBRParams::* member)
{
    Binding b;
    b.asBoolMember = member;
    return b;
}

/// A rotation the file states in turns, stored as the direction it names.
Binding bindRotationTurns(float OpenPBRParams::* cosMember, float OpenPBRParams::* sinMember)
{
    Binding b;
    b.asFloatMember = cosMember;
    b.asSinMember = sinMember;
    return b;
}

/// Texture-only: the input carries a map and no value this struct can hold.
Binding bindTextureOnly(int slot)
{
    Binding b;
    b.textureSlot = slot;
    return b;
}

void writeBinding(OpenPBRParams& p, const Binding& b, const mx::ValuePtr& v)
{
    if (b.asSinMember)
    {
        // Turns, not radians: Standard Surface documents specular_rotation and
        // coat_rotation over [0,1] for the full circle.
        const float radians = asFloat(v, 0.0f) * 2.0f * 3.14159265358979323846f;
        p.*(b.asFloatMember) = std::cos(radians);
        p.*(b.asSinMember) = std::sin(radians);
        return;
    }
    if (b.asFloatMember)
    {
        p.*(b.asFloatMember) = asFloat(v, p.*(b.asFloatMember));
        return;
    }
    if (b.asColorMember)
    {
        p.*(b.asColorMember) = asColor(v, p.*(b.asColorMember));
        return;
    }
    if (b.asBoolMember)
    {
        p.*(b.asBoolMember) = asBool(v, false) ? 1u : 0u;
    }
}

const std::unordered_map<std::string, Binding>& openPbrBindings()
{
    static const std::unordered_map<std::string, Binding> kTable = {
        { "base_weight", bindFloat(&OpenPBRParams::base_weight) },
        { "base_color", bindColor(&OpenPBRParams::base_color, OPENPBR_TEX_BASE_COLOR) },
        { "base_diffuse_roughness", bindFloat(&OpenPBRParams::base_diffuse_roughness) },
        { "base_metalness", bindFloat(&OpenPBRParams::base_metalness, OPENPBR_TEX_BASE_METALNESS) },
        { "specular_weight", bindFloat(&OpenPBRParams::specular_weight) },
        { "specular_color", bindColor(&OpenPBRParams::specular_color, OPENPBR_TEX_SPECULAR_COLOR) },
        { "specular_roughness", bindFloat(&OpenPBRParams::specular_roughness, OPENPBR_TEX_SPECULAR_ROUGHNESS) },
        { "specular_roughness_anisotropy",
          bindFloat(&OpenPBRParams::specular_roughness_anisotropy, OPENPBR_TEX_SPECULAR_ANISOTROPY) },
        { "specular_ior", bindFloat(&OpenPBRParams::specular_ior) },
        { "coat_weight", bindFloat(&OpenPBRParams::coat_weight, OPENPBR_TEX_COAT_WEIGHT) },
        { "coat_color", bindColor(&OpenPBRParams::coat_color, OPENPBR_TEX_COAT_COLOR) },
        { "coat_roughness", bindFloat(&OpenPBRParams::coat_roughness, OPENPBR_TEX_COAT_ROUGHNESS) },
        { "coat_roughness_anisotropy", bindFloat(&OpenPBRParams::coat_roughness_anisotropy) },
        { "coat_ior", bindFloat(&OpenPBRParams::coat_ior) },
        { "coat_darkening", bindFloat(&OpenPBRParams::coat_darkening) },
        { "fuzz_weight", bindFloat(&OpenPBRParams::fuzz_weight, OPENPBR_TEX_FUZZ_WEIGHT) },
        { "fuzz_color", bindColor(&OpenPBRParams::fuzz_color, OPENPBR_TEX_FUZZ_COLOR) },
        { "fuzz_roughness", bindFloat(&OpenPBRParams::fuzz_roughness, OPENPBR_TEX_FUZZ_ROUGHNESS) },
        { "transmission_weight", bindFloat(&OpenPBRParams::transmission_weight) },
        { "transmission_color", bindColor(&OpenPBRParams::transmission_color, OPENPBR_TEX_TRANSMISSION_COLOR) },
        { "transmission_depth", bindFloat(&OpenPBRParams::transmission_depth) },
        { "transmission_scatter", bindColor(&OpenPBRParams::transmission_scatter) },
        { "transmission_scatter_anisotropy", bindFloat(&OpenPBRParams::transmission_scatter_anisotropy) },
        { "transmission_dispersion_scale", bindFloat(&OpenPBRParams::transmission_dispersion_scale) },
        { "transmission_dispersion_abbe_number", bindFloat(&OpenPBRParams::transmission_dispersion_abbe_number) },
        { "subsurface_weight", bindFloat(&OpenPBRParams::subsurface_weight, OPENPBR_TEX_SUBSURFACE_WEIGHT) },
        { "subsurface_color", bindColor(&OpenPBRParams::subsurface_color, OPENPBR_TEX_SUBSURFACE_COLOR) },
        { "subsurface_radius", bindFloat(&OpenPBRParams::subsurface_radius) },
        { "subsurface_radius_scale", bindColor(&OpenPBRParams::subsurface_radius_scale, OPENPBR_TEX_SUBSURFACE_RADIUS) },
        { "subsurface_scatter_anisotropy", bindFloat(&OpenPBRParams::subsurface_scatter_anisotropy) },
        { "thin_film_weight", bindFloat(&OpenPBRParams::thin_film_weight) },
        { "thin_film_thickness", bindFloat(&OpenPBRParams::thin_film_thickness) },
        { "thin_film_ior", bindFloat(&OpenPBRParams::thin_film_ior) },
        { "emission_luminance", bindFloat(&OpenPBRParams::emission_luminance) },
        { "emission_color", bindColor(&OpenPBRParams::emission_color, OPENPBR_TEX_EMISSION_COLOR) },
        { "geometry_opacity", bindFloat(&OpenPBRParams::geometry_opacity, OPENPBR_TEX_GEOMETRY_OPACITY) },
        { "geometry_thin_walled", bindBool(&OpenPBRParams::geometry_thin_walled) },
        { "geometry_normal", bindTextureOnly(OPENPBR_TEX_GEOMETRY_NORMAL) },
        { "geometry_coat_normal", bindTextureOnly(OPENPBR_TEX_GEOMETRY_COAT_NORMAL) },
    };
    return kTable;
}

const std::unordered_map<std::string, Binding>& standardSurfaceBindings()
{
    static const std::unordered_map<std::string, Binding> kTable = {
        { "base", bindFloat(&OpenPBRParams::base_weight) },
        { "base_color", bindColor(&OpenPBRParams::base_color, OPENPBR_TEX_BASE_COLOR) },
        { "diffuse_roughness", bindFloat(&OpenPBRParams::base_diffuse_roughness) },
        { "metalness", bindFloat(&OpenPBRParams::base_metalness, OPENPBR_TEX_BASE_METALNESS) },
        { "specular", bindFloat(&OpenPBRParams::specular_weight) },
        { "specular_color", bindColor(&OpenPBRParams::specular_color, OPENPBR_TEX_SPECULAR_COLOR) },
        { "specular_roughness", bindFloat(&OpenPBRParams::specular_roughness, OPENPBR_TEX_SPECULAR_ROUGHNESS) },
        { "specular_IOR", bindFloat(&OpenPBRParams::specular_ior) },
        { "specular_anisotropy",
          bindFloat(&OpenPBRParams::specular_roughness_anisotropy, OPENPBR_TEX_SPECULAR_ANISOTROPY) },
        { "specular_rotation", bindRotationTurns(&OpenPBRParams::specular_anisotropy_rotation_cos,
                                                 &OpenPBRParams::specular_anisotropy_rotation_sin) },
        { "transmission", bindFloat(&OpenPBRParams::transmission_weight) },
        { "transmission_color", bindColor(&OpenPBRParams::transmission_color, OPENPBR_TEX_TRANSMISSION_COLOR) },
        { "transmission_depth", bindFloat(&OpenPBRParams::transmission_depth) },
        { "transmission_scatter", bindColor(&OpenPBRParams::transmission_scatter) },
        { "transmission_scatter_anisotropy", bindFloat(&OpenPBRParams::transmission_scatter_anisotropy) },
        { "transmission_dispersion", bindFloat(&OpenPBRParams::transmission_dispersion_scale) },
        { "subsurface", bindFloat(&OpenPBRParams::subsurface_weight, OPENPBR_TEX_SUBSURFACE_WEIGHT) },
        { "subsurface_color", bindColor(&OpenPBRParams::subsurface_color, OPENPBR_TEX_SUBSURFACE_COLOR) },
        { "subsurface_anisotropy", bindFloat(&OpenPBRParams::subsurface_scatter_anisotropy) },
        // Only reached when a map drives it; the constant form is combined with
        // subsurface_scale above, which no per-input binding can do.
        { "subsurface_radius", bindColor(&OpenPBRParams::subsurface_radius_scale, OPENPBR_TEX_SUBSURFACE_RADIUS) },
        { "sheen", bindFloat(&OpenPBRParams::fuzz_weight, OPENPBR_TEX_FUZZ_WEIGHT) },
        { "sheen_color", bindColor(&OpenPBRParams::fuzz_color, OPENPBR_TEX_FUZZ_COLOR) },
        { "sheen_roughness", bindFloat(&OpenPBRParams::fuzz_roughness, OPENPBR_TEX_FUZZ_ROUGHNESS) },
        { "coat", bindFloat(&OpenPBRParams::coat_weight, OPENPBR_TEX_COAT_WEIGHT) },
        { "coat_color", bindColor(&OpenPBRParams::coat_color, OPENPBR_TEX_COAT_COLOR) },
        { "coat_roughness", bindFloat(&OpenPBRParams::coat_roughness, OPENPBR_TEX_COAT_ROUGHNESS) },
        { "coat_anisotropy", bindFloat(&OpenPBRParams::coat_roughness_anisotropy) },
        { "coat_rotation", bindRotationTurns(&OpenPBRParams::coat_anisotropy_rotation_cos,
                                             &OpenPBRParams::coat_anisotropy_rotation_sin) },
        { "coat_IOR", bindFloat(&OpenPBRParams::coat_ior) },
        { "thin_film_thickness", bindFloat(&OpenPBRParams::thin_film_thickness) },
        { "thin_film_IOR", bindFloat(&OpenPBRParams::thin_film_ior) },
        { "emission", bindFloat(&OpenPBRParams::emission_luminance) },
        { "emission_color", bindColor(&OpenPBRParams::emission_color, OPENPBR_TEX_EMISSION_COLOR) },
        { "opacity", bindFloat(&OpenPBRParams::geometry_opacity, OPENPBR_TEX_GEOMETRY_OPACITY) },
        { "thin_walled", bindBool(&OpenPBRParams::geometry_thin_walled) },
        { "normal", bindTextureOnly(OPENPBR_TEX_GEOMETRY_NORMAL) },
        { "coat_normal", bindTextureOnly(OPENPBR_TEX_GEOMETRY_COAT_NORMAL) },
    };
    return kTable;
}


} // namespace

MaterialXDocumentData loadMaterialXDocument(const std::string& path)
{
    MaterialXDocumentData result;

    const mx::DocumentPtr doc = mx::createDocument();

    // The data library first. Without it a document that says
    // <standard_surface> refers to a nodedef that does not exist, and MaterialX
    // will read the file without complaint and hand back a node with no type.
    const std::string libRoot = oka::resolveResourcePath("materialx/libraries");
    if (!libRoot.empty())
    {
        try
        {
            mx::FileSearchPath searchPath;
            searchPath.append(mx::FilePath(libRoot).getParentPath());
            mx::loadLibraries({ mx::FilePath(libRoot).getBaseName() }, searchPath, doc);
        }
        catch (const std::exception& e)
        {
            STRELKA_WARNING("MaterialX data library at {} did not load: {}", libRoot, e.what());
        }
    }
    else
    {
        STRELKA_WARNING("MaterialX data library not found; nodedef defaults will be missing");
    }

    const mx::FilePath docPath(path);
    try
    {
        mx::FileSearchPath searchPath;
        searchPath.append(docPath.getParentPath());
        mx::readFromXmlFile(doc, docPath, searchPath);
    }
    catch (const std::exception& e)
    {
        STRELKA_ERROR("MaterialX {}: {}", path, e.what());
        return result;
    }

    const mx::FilePath docDir = docPath.getParentPath();

    for (const mx::NodePtr& materialNode : doc->getMaterialNodes())
    {
        const std::vector<mx::NodePtr> shaders = mx::getShaderNodes(materialNode);
        if (shaders.empty())
        {
            STRELKA_WARNING("MaterialX {}: material '{}' has no surface shader", path, materialNode->getName());
            continue;
        }

        const mx::NodePtr& shader = shaders.front();
        const std::string category = shader->getCategory();
        const std::unordered_map<std::string, Binding>* table = nullptr;
        if (category == "open_pbr_surface")
        {
            table = &openPbrBindings();
        }
        else if (category == "standard_surface")
        {
            table = &standardSurfaceBindings();
        }
        else
        {
            STRELKA_WARNING("MaterialX {}: '{}' is a <{}>, which this loader does not map", path,
                            materialNode->getName(), category);
            continue;
        }

        MaterialXMaterial out;
        out.name = materialNode->getName();
        out.params = openpbr_make_default_params();

        // standard_surface splits the subsurface mean free path into a colour
        // and a scale; OpenPBR keeps a length and a normalised tint. Collected
        // here because the two inputs have to be combined, which no per-input
        // binding can do.
        OpenPBRColor ssRadius{ 1.0f, 1.0f, 1.0f };
        float ssScale = 1.0f;
        bool sawRadius = false;
        bool sawScale = false;
        float thinFilmThickness = 0.0f;
        bool sawThinFilm = false;

        for (const mx::InputPtr& input : shader->getInputs())
        {
            const std::string name = input->getName();
            const Resolved r = resolveInput(input, docDir);

            if (!r.unsupportedCategory.empty())
            {
                out.unsupported.push_back(name + "<-" + r.unsupportedCategory);
            }

            if (category == "standard_surface")
            {
                if (name == "subsurface_radius" && r.isConstant)
                {
                    ssRadius = asColor(r.value, ssRadius);
                    sawRadius = true;
                    continue;
                }
                if (name == "subsurface_scale" && r.isConstant)
                {
                    ssScale = asFloat(r.value, ssScale);
                    sawScale = true;
                    continue;
                }
                if (name == "thin_film_thickness" && r.isConstant)
                {
                    thinFilmThickness = asFloat(r.value, 0.0f);
                    sawThinFilm = true;
                }
            }

            const auto it = table->find(name);
            if (it == table->end())
            {
                continue; // an input this model has and OpenPBR does not
            }
            const Binding& binding = it->second;

            if (!r.texture.empty() && binding.textureSlot >= 0)
            {
                out.texPaths[(size_t)binding.textureSlot] = r.texture;
            }
            else if (!r.texture.empty())
            {
                out.unsupported.push_back(name + "<-image(no slot)");
            }

            if (r.isConstant)
            {
                writeBinding(out.params, binding, r.value);
            }
        }

        if (category == "standard_surface")
        {
            if (sawRadius || sawScale)
            {
                // The largest channel keeps the world-space length; the tint is
                // what is left. Same split openpbr_from_gltf.h makes.
                const float maxChannel = std::max({ ssRadius.r, ssRadius.g, ssRadius.b });
                if (maxChannel > 0.0f)
                {
                    out.params.subsurface_radius = ssScale * maxChannel;
                    out.params.subsurface_radius_scale =
                        OpenPBRColor{ ssRadius.r / maxChannel, ssRadius.g / maxChannel, ssRadius.b / maxChannel };
                }
            }
            if (sawThinFilm)
            {
                // Standard Surface has no thin_film weight: the film is present
                // when it has a thickness. OpenPBR states the two separately, so
                // the weight has to be derived or every material gets a film.
                out.params.thin_film_weight = (thinFilmThickness > 0.0f) ? 1.0f : 0.0f;
            }
        }

        result.materials.push_back(std::move(out));
    }

    // The look, which is how a MaterialX scene actually binds: by geometry, not
    // by material name.
    for (const mx::LookPtr& look : doc->getLooks())
    {
        for (const mx::MaterialAssignPtr& assign : look->getMaterialAssigns())
        {
            std::string geom = assign->getGeom();
            // Geometry names are paths; a single leading separator is the whole
            // of what these documents use.
            if (!geom.empty() && geom.front() == '/')
            {
                geom.erase(0, 1);
            }
            if (geom.empty() || assign->getMaterial().empty())
            {
                continue;
            }
            result.assignments.push_back({ geom, assign->getMaterial() });
        }
    }

    return result;
}

int applyMaterialXDocument(Scene& scene, const std::string& path)
{
    const MaterialXDocumentData doc = loadMaterialXDocument(path);
    const std::vector<MaterialXMaterial>& materials = doc.materials;
    if (materials.empty())
    {
        return 0;
    }

    std::vector<Scene::MaterialDescription>& descs = scene.getMaterials();
    int applied = 0;
    for (const MaterialXMaterial& m : materials)
    {
        // `M_Bishop_B` should find a glTF material called `Bishop_B`. The Open
        // Chess Set names its materials that way and its geometry the other, and
        // an exact-match-only rule would bind nothing while looking like it had
        // read the file.
        std::string alias = m.name;
        if (alias.starts_with("M_"))
        {
            alias = alias.substr(2);
        }

        bool matched = false;
        for (Scene::MaterialDescription& desc : descs)
        {
            if (desc.name != m.name && desc.name != alias)
            {
                continue;
            }
            desc.openpbr = m.params;
            desc.openpbrTexPaths = m.texPaths;
            desc.params.material_type = MATERIAL_TYPE_OPENPBR;
            matched = true;
            ++applied;
        }
        if (!matched)
        {
            // Only worth saying when nothing else will bind this material. A
            // document that assigns through a <look> -- which is the normal way
            // to build a MaterialX scene -- has no reason to name its materials
            // after the glTF ones, and warning there made the Open Chess Set
            // print fifteen complaints about a file it went on to load
            // correctly.
            const bool boundByLook =
                std::ranges::any_of(doc.assignments, [&](const MaterialXAssignment& a) { return a.material == m.name; });
            if (!boundByLook)
            {
                STRELKA_WARNING("MaterialX {}: no scene material named '{}' (or '{}'), and no look assigns it", path,
                                m.name, alias);
            }
        }
        if (!m.unsupported.empty())
        {
            std::string joined;
            for (const std::string& u : m.unsupported)
            {
                joined += (joined.empty() ? "" : ", ") + u;
            }
            STRELKA_WARNING("MaterialX {}: '{}' has inputs this loader cannot express: {}", path, m.name, joined);
        }
    }
    // Then the look. A material bound this way gets its own scene material and
    // is pointed at from the instances of the named node, because the geometry a
    // look assigns to usually shares one placeholder material with everything
    // else -- rewriting that in place would repaint the whole board.
    for (const MaterialXAssignment& assign : doc.assignments)
    {
        const auto found =
            std::ranges::find_if(materials, [&](const MaterialXMaterial& m) { return m.name == assign.material; });
        if (found == materials.end())
        {
            STRELKA_WARNING("MaterialX {}: look assigns '{}' to '{}', which the document does not define", path,
                            assign.material, assign.geom);
            continue;
        }

        Scene::MaterialDescription desc{};
        desc.name = assign.material + "@" + assign.geom;
        desc.params = MaterialParams{};
        desc.params.material_type = MATERIAL_TYPE_OPENPBR;
        desc.params.base_color = { found->params.base_color.r, found->params.base_color.g, found->params.base_color.b };
        desc.params.alpha_mode = ALPHA_MODE_OPAQUE;
        desc.params.base_color_alpha = 1.0f;
        desc.params.uv_scale_x = 1.0f;
        desc.params.uv_scale_y = 1.0f;
        desc.openpbr = found->params;
        desc.openpbrTexPaths = found->texPaths;
        const uint32_t materialId = scene.addMaterial(desc);

        int instances = 0;
        for (const Scene::Node& node : scene.getNodes())
        {
            if (node.name != assign.geom)
            {
                continue;
            }
            for (const uint32_t instId : node.instanceIds)
            {
                if (instId < scene.getInstances().size())
                {
                    scene.getInstances()[instId].mMaterialId = materialId;
                    ++instances;
                }
            }
        }
        if (instances == 0)
        {
            STRELKA_WARNING(
                "MaterialX {}: look assigns to '{}', which no node in the scene is called", path, assign.geom);
        }
        else
        {
            applied += instances;
        }
    }

    STRELKA_INFO("MaterialX {}: {} material(s) defined, {} binding(s) applied", path, materials.size(), applied);
    return applied;
}

} // namespace oka::mtlx
