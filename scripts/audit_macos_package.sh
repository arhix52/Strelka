#!/usr/bin/env bash
# Reject Metal profiling data and Strelka shader sources in a distributable app.
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "Usage: $0 <Strelka.app|package-root> [source-root]" >&2
    exit 2
fi

TARGET="$1"
SOURCE_ROOT="${2:-}"

fail()
{
    echo "error: package audit: $*" >&2
    exit 1
}

[[ -e "${TARGET}" ]] || fail "${TARGET} does not exist"
command -v xcrun >/dev/null 2>&1 || fail "xcrun is unavailable"

if [[ "${TARGET}" == *.app ]]; then
    APP="${TARGET}"
    SHADER_DIRS=("${APP}/Contents/Resources/metal/shaders")
else
    APP="${TARGET}/Strelka.app"
    SHADER_DIRS=(
        "${TARGET}/metal/shaders"
        "${APP}/Contents/Resources/metal/shaders"
    )
fi

[[ -d "${APP}" ]] || fail "${APP} is missing"

while IFS= read -r artifact; do
    fail "profiling artifact is packaged: ${artifact}"
done < <(find "${TARGET}" \
    \( -name '*.air' -o -name '*.metallibsym' -o -name '*.dSYM' -o -name '*.gputrace' \) \
    -print)

EXPECTED=(fullScreen skinning tonemapper wavefront)
for shader_dir in "${SHADER_DIRS[@]}"; do
    [[ -d "${shader_dir}" ]] || fail "shader directory is missing: ${shader_dir}"

    for path in "${shader_dir}"/*; do
        [[ -e "${path}" ]] || continue
        [[ -f "${path}" && "${path}" == *.metallib ]] || fail "unexpected shader resource: ${path}"
    done

    for shader in "${EXPECTED[@]}"; do
        lib="${shader_dir}/${shader}.metallib"
        [[ -s "${lib}" ]] || fail "${lib} is missing or empty"
    done

    for lib in "${shader_dir}"/*.metallib; do
        if ! validation="$(xcrun metallib --app-store-validate "${lib}" 2>&1)"; then
            fail "${lib} failed App Store validation: ${validation}"
        fi

        sections="$(xcrun metal-objdump --section-headers "${lib}")"
        if grep -Eq '(^|[[:space:]])(SOURCES|DYNAMIC_HEADER|REFLECTION_LIST)([[:space:]]|$)' <<<"${sections}"; then
            fail "${lib} contains Metal source or profiling sections"
        fi

        shader_strings="$(strings -a "${lib}")"
        if [[ -n "${SOURCE_ROOT}" ]] && grep -Fq "${SOURCE_ROOT}" <<<"${shader_strings}"; then
            fail "${lib} contains the source-tree path ${SOURCE_ROOT}"
        fi
        if grep -Eq '(/Users/[^/]+|/home/[^/]+|[A-Za-z]:\\Users\\[^\\]+).*(Strelka|src[/\\]shaders)' \
            <<<"${shader_strings}"; then
            fail "${lib} contains a developer source path"
        fi
    done
done

echo "Metal package audit OK: ${TARGET}"
