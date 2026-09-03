#ifndef STRELKA_MATERIAL_OPENPBR_CUDA_VEC_H
#define STRELKA_MATERIAL_OPENPBR_CUDA_VEC_H

// The vec2/vec3/vec4 openpbr's CUDA interop layer should have shipped.
//
// `interop/openpbr_interop_cuda.h` aliases them to CUDA's `float2/float3/float4`
// unless OPENPBR_USE_CUSTOM_VEC_TYPES is set. That alias cannot work, and this
// header exists because the failure is not subtle once you look for it: openpbr
// is written against GLSL and GLM vectors, and CUDA's are bare aggregates.
// Compiling `openpbr.h` under nvcc with the shipped alias produces, in order,
//
//     vec3(a, b, c)      no matching constructor -- float3 is an aggregate
//     v[0]               no operator[]
//     c.r, c.g, c.b      no such members
//     a != b, a > b      no comparison operators
//     exp(v), max(v, s)  no GLSL-named componentwise math
//
// so what follows is not a convenience layer: it is the missing half of the
// interop. It is included by openpbr_shim.h on the CUDA branch only. Metal and
// the host both have a vector type that satisfies openpbr already (MSL's
// `float3`, GLM's `glm::vec3`), and both take the interop's own path.
//
// Two consequences worth stating, because they leak into calling code:
//
//  * `vec3` is a *distinct type* from `float3` here, not an alias. Conversion
//    runs implicitly in both directions, so expressions mix freely, but a
//    non-const `vec3&` out-parameter will not bind to a `float3` lvalue.
//    openpbr_bsdf_sample() in openpbr_bridge.h is the one place that matters.
//
//  * The x/y/z and r/g/b spellings are one union, matching GLM. The layout is
//    three floats, same as `float3`; nothing here changes the size or alignment
//    of anything the host uploads.
//
// The comparison operators follow the semantics openpbr's own CUDA interop
// documents for its `equal`/`greaterThan` macros: they reduce over all
// components, because openpbr only ever uses them inside `all()`/`any()`, which
// that layer defines as identities.

#include <vector_types.h>

#define OPENPBR_USE_CUSTOM_VEC_TYPES 1

// __host__ __device__ throughout: the tables are constexpr globals the host
// compiler still has to parse, and the unit tests may evaluate a constructor at
// compile time on either side.
#define STRELKA_OPENPBR_VEC_FUNC __host__ __device__

struct vec2
{
    union
    {
        struct
        {
            float x, y;
        };
        struct
        {
            float r, g;
        };
    };

    vec2() = default;
    STRELKA_OPENPBR_VEC_FUNC constexpr vec2(float a, float b) : x(a), y(b) {}
    STRELKA_OPENPBR_VEC_FUNC explicit constexpr vec2(float s) : x(s), y(s) {}
    STRELKA_OPENPBR_VEC_FUNC constexpr vec2(const float2& v) : x(v.x), y(v.y) {}
    STRELKA_OPENPBR_VEC_FUNC constexpr operator float2() const { return float2{ x, y }; }
    STRELKA_OPENPBR_VEC_FUNC float& operator[](int i) { return (&x)[i]; }
    STRELKA_OPENPBR_VEC_FUNC const float& operator[](int i) const { return (&x)[i]; }
};

struct vec3
{
    union
    {
        struct
        {
            float x, y, z;
        };
        struct
        {
            float r, g, b;
        };
    };

    vec3() = default;
    STRELKA_OPENPBR_VEC_FUNC constexpr vec3(float a, float b_, float c) : x(a), y(b_), z(c) {}
    STRELKA_OPENPBR_VEC_FUNC explicit constexpr vec3(float s) : x(s), y(s), z(s) {}
    STRELKA_OPENPBR_VEC_FUNC constexpr vec3(const float3& v) : x(v.x), y(v.y), z(v.z) {}
    STRELKA_OPENPBR_VEC_FUNC constexpr operator float3() const { return float3{ x, y, z }; }
    STRELKA_OPENPBR_VEC_FUNC float& operator[](int i) { return (&x)[i]; }
    STRELKA_OPENPBR_VEC_FUNC const float& operator[](int i) const { return (&x)[i]; }
};

struct vec4
{
    union
    {
        struct
        {
            float x, y, z, w;
        };
        struct
        {
            float r, g, b, a;
        };
    };

    vec4() = default;
    STRELKA_OPENPBR_VEC_FUNC constexpr vec4(float a_, float b_, float c, float d) : x(a_), y(b_), z(c), w(d) {}
    STRELKA_OPENPBR_VEC_FUNC explicit constexpr vec4(float s) : x(s), y(s), z(s), w(s) {}
    STRELKA_OPENPBR_VEC_FUNC constexpr vec4(const vec3& v, float d) : x(v.x), y(v.y), z(v.z), w(d) {}
    STRELKA_OPENPBR_VEC_FUNC constexpr vec4(const float4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
    STRELKA_OPENPBR_VEC_FUNC constexpr operator float4() const { return float4{ x, y, z, w }; }
    STRELKA_OPENPBR_VEC_FUNC float& operator[](int i) { return (&x)[i]; }
    STRELKA_OPENPBR_VEC_FUNC const float& operator[](int i) const { return (&x)[i]; }
};

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------
//
// Written through operator[] and a fixed-count loop rather than component by
// component: the loop bound is a literal, so every one of these unrolls, and the
// macro then covers all three widths without twelve near-identical bodies to
// keep in step.

#define STRELKA_OPENPBR_VEC_BINOP(T, N, op)                                                                            \
    STRELKA_OPENPBR_VEC_FUNC inline T operator op(const T a, const T b)                                                \
    {                                                                                                                  \
        T r;                                                                                                           \
        for (int i = 0; i < N; ++i)                                                                                    \
            r[i] = a[i] op b[i];                                                                                       \
        return r;                                                                                                      \
    }                                                                                                                  \
    STRELKA_OPENPBR_VEC_FUNC inline T operator op(const T a, float s)                                                  \
    {                                                                                                                  \
        T r;                                                                                                           \
        for (int i = 0; i < N; ++i)                                                                                    \
            r[i] = a[i] op s;                                                                                          \
        return r;                                                                                                      \
    }                                                                                                                  \
    STRELKA_OPENPBR_VEC_FUNC inline T operator op(float s, const T a)                                                  \
    {                                                                                                                  \
        T r;                                                                                                           \
        for (int i = 0; i < N; ++i)                                                                                    \
            r[i] = s op a[i];                                                                                          \
        return r;                                                                                                      \
    }                                                                                                                  \
    STRELKA_OPENPBR_VEC_FUNC inline T& operator op##=(T& a, const T b)                                                 \
    {                                                                                                                  \
        a = a op b;                                                                                                    \
        return a;                                                                                                      \
    }                                                                                                                  \
    STRELKA_OPENPBR_VEC_FUNC inline T& operator op##=(T& a, float s)                                                   \
    {                                                                                                                  \
        a = a op s;                                                                                                    \
        return a;                                                                                                      \
    }

STRELKA_OPENPBR_VEC_BINOP(vec2, 2, +)
STRELKA_OPENPBR_VEC_BINOP(vec2, 2, -)
STRELKA_OPENPBR_VEC_BINOP(vec2, 2, *)
STRELKA_OPENPBR_VEC_BINOP(vec2, 2, /)
STRELKA_OPENPBR_VEC_BINOP(vec3, 3, +)
STRELKA_OPENPBR_VEC_BINOP(vec3, 3, -)
STRELKA_OPENPBR_VEC_BINOP(vec3, 3, *)
STRELKA_OPENPBR_VEC_BINOP(vec3, 3, /)
STRELKA_OPENPBR_VEC_BINOP(vec4, 4, +)
STRELKA_OPENPBR_VEC_BINOP(vec4, 4, -)
STRELKA_OPENPBR_VEC_BINOP(vec4, 4, *)
STRELKA_OPENPBR_VEC_BINOP(vec4, 4, /)
#undef STRELKA_OPENPBR_VEC_BINOP

STRELKA_OPENPBR_VEC_FUNC inline vec3 operator-(const vec3 a)
{
    return vec3(-a.x, -a.y, -a.z);
}

// All-component reduction -- see the header note.
#define STRELKA_OPENPBR_VEC_CMP(T, N, op)                                                                              \
    STRELKA_OPENPBR_VEC_FUNC inline bool operator op(const T a, const T b)                                             \
    {                                                                                                                  \
        bool r = true;                                                                                                 \
        for (int i = 0; i < N; ++i)                                                                                    \
            r = r && (a[i] op b[i]);                                                                                   \
        return r;                                                                                                      \
    }

STRELKA_OPENPBR_VEC_CMP(vec2, 2, ==)
STRELKA_OPENPBR_VEC_CMP(vec2, 2, >)
STRELKA_OPENPBR_VEC_CMP(vec3, 3, ==)
STRELKA_OPENPBR_VEC_CMP(vec3, 3, >)
STRELKA_OPENPBR_VEC_CMP(vec4, 4, ==)
STRELKA_OPENPBR_VEC_CMP(vec4, 4, >)
#undef STRELKA_OPENPBR_VEC_CMP

// `!=` is the negation of the all-component `==`, i.e. "differs in at least one
// component". That is what openpbr's `any(notEqual(a, b))` means, and the CUDA
// interop's own comment says the same.
STRELKA_OPENPBR_VEC_FUNC inline bool operator!=(const vec2 a, const vec2 b)
{
    return !(a == b);
}
STRELKA_OPENPBR_VEC_FUNC inline bool operator!=(const vec3 a, const vec3 b)
{
    return !(a == b);
}
STRELKA_OPENPBR_VEC_FUNC inline bool operator!=(const vec4 a, const vec4 b)
{
    return !(a == b);
}

// ---------------------------------------------------------------------------
// GLSL-named math
// ---------------------------------------------------------------------------
//
// sutil's vec_math.h has most of these for float3 under CUDA's names (fmaxf,
// expf, ...). openpbr calls them by their GLSL names on vector arguments, so
// these are the spellings it needs, defined on the vec types only -- nothing
// here overloads anything that already takes a float3.
//
// This is exactly the set the tree needs and not one function more. An earlier
// version also carried componentwise abs/acos/cos/sin/floor, a second pow and
// min, both clamps, a vector smoothstep, `>=` and unary minus; openpbr calls
// those on scalars, where CUDA's own overloads answer, and none of them is
// reachable from here.
//
// Two things this set rests on, both of which cost a measurement to find:
//
//  * The operators above and the geometric functions below cannot go, even
//    though sutil has float3 versions of every one. Theirs return `float3`, and
//    openpbr feeds the result straight back into an expression that also holds a
//    `vec3` -- `mix()` inside a ternary, in openpbr_bsdf.h. Both conversions
//    then apply and the operands are ambiguous. Keeping the result in `vec3` is
//    what these are for, not the arithmetic.
//
//  * **Compiling is not the test.** Removing dot/length/normalize for `vec2`
//    builds clean: resolution simply falls through to sutil's float2 versions,
//    which compute the same quantities by a different expression. Under
//    --use_fast_math that is not the same number, and the chess set came back
//    with 1563 differing bytes -- about 500 pixels -- from a change that looked
//    like pure deletion. Anything removed from this header has to be graded on a
//    render, not on a build.

#define STRELKA_OPENPBR_VEC_CWISE(name, f)                                                                             \
    STRELKA_OPENPBR_VEC_FUNC inline vec2 name(const vec2 v)                                                            \
    {                                                                                                                  \
        return vec2(f(v.x), f(v.y));                                                                                   \
    }                                                                                                                  \
    STRELKA_OPENPBR_VEC_FUNC inline vec3 name(const vec3 v)                                                            \
    {                                                                                                                  \
        return vec3(f(v.x), f(v.y), f(v.z));                                                                           \
    }

STRELKA_OPENPBR_VEC_CWISE(exp, ::expf)
STRELKA_OPENPBR_VEC_CWISE(log, ::logf)
STRELKA_OPENPBR_VEC_CWISE(sqrt, ::sqrtf)
#undef STRELKA_OPENPBR_VEC_CWISE

STRELKA_OPENPBR_VEC_FUNC inline vec3 pow(const vec3 a, const vec3 b)
{
    return vec3(::powf(a.x, b.x), ::powf(a.y, b.y), ::powf(a.z, b.z));
}
STRELKA_OPENPBR_VEC_FUNC inline vec3 max(const vec3 a, const vec3 b)
{
    return vec3(::fmaxf(a.x, b.x), ::fmaxf(a.y, b.y), ::fmaxf(a.z, b.z));
}
STRELKA_OPENPBR_VEC_FUNC inline vec3 max(const vec3 a, float s)
{
    return vec3(::fmaxf(a.x, s), ::fmaxf(a.y, s), ::fmaxf(a.z, s));
}
STRELKA_OPENPBR_VEC_FUNC inline vec3 min(const vec3 a, const vec3 b)
{
    return vec3(::fminf(a.x, b.x), ::fminf(a.y, b.y), ::fminf(a.z, b.z));
}
STRELKA_OPENPBR_VEC_FUNC inline vec2 max(const vec2 a, const vec2 b)
{
    return vec2(::fmaxf(a.x, b.x), ::fmaxf(a.y, b.y));
}

// GLSL argument order (edge0, edge1, x), which is not the CUDA one.
STRELKA_OPENPBR_VEC_FUNC inline float smoothstep(float e0, float e1, float x)
{
    const float t = ::fminf(::fmaxf((x - e0) / (e1 - e0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

STRELKA_OPENPBR_VEC_FUNC inline float dot(const vec3 a, const vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
STRELKA_OPENPBR_VEC_FUNC inline float length(const vec3 v)
{
    return ::sqrtf(dot(v, v));
}
STRELKA_OPENPBR_VEC_FUNC inline vec3 normalize(const vec3 v)
{
    return v * ::rsqrtf(dot(v, v));
}
STRELKA_OPENPBR_VEC_FUNC inline vec3 cross(const vec3 a, const vec3 b)
{
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
STRELKA_OPENPBR_VEC_FUNC inline float dot(const vec2 a, const vec2 b)
{
    return a.x * b.x + a.y * b.y;
}
STRELKA_OPENPBR_VEC_FUNC inline float length(const vec2 v)
{
    return ::sqrtf(dot(v, v));
}
STRELKA_OPENPBR_VEC_FUNC inline vec2 normalize(const vec2 v)
{
    return v * ::rsqrtf(dot(v, v));
}
// GLSL's reflect: `i` points *at* the surface, so the sign is not the one
// sutil's reflect(i, n) uses for an outgoing direction.
STRELKA_OPENPBR_VEC_FUNC inline vec3 reflect(const vec3 i, const vec3 n)
{
    return i - n * (2.0f * dot(n, i));
}

#endif // STRELKA_MATERIAL_OPENPBR_CUDA_VEC_H
