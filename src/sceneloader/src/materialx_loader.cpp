#include <strelka/sceneloader/materialx_loader.h>

#include <log.h>
#include <paths.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Look.h>
#include <MaterialXCore/Material.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

#include <algorithm>
#include <filesystem>
#include <ranges>
#include <cctype>
#include <cmath>
#include <unordered_map>

namespace mx = MaterialX;

namespace oka::mtlx
{
namespace
{

/// The UV placement an image asks for, already in the form the shader applies:
///
///     tuv = rotate_ccw(uv * scale, rotation) + offset
///
/// which is one transform per *material*, not per slot -- OpenPBRParams carries
/// a single uv_scale/uv_rotation/uv_offset that every map of that material is
/// sampled through. Two maps placed differently cannot both be honoured, so the
/// second one is reported rather than quietly overwriting the first.
struct UvPlacement
{
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float rotation = 0.0f; // radians, counter-clockwise, the shader's convention
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    bool stated = false; // the document asked for something other than identity
};

bool placementsAgree(const UvPlacement& a, const UvPlacement& b)
{
    const auto same = [](float x, float y) { return std::fabs(x - y) <= 1e-6f; };
    return same(a.scaleX, b.scaleX) && same(a.scaleY, b.scaleY) && same(a.rotation, b.rotation) &&
           same(a.offsetX, b.offsetX) && same(a.offsetY, b.offsetY);
}

/// What an input resolved to.
struct Resolved
{
    bool isConstant = false;
    mx::ValuePtr value;
    std::string texture; // absolute path, empty when none
    std::string colorSpace; // verbatim from the document, empty when unstated
    UvPlacement placement;
    std::string unsupportedCategory;
};

/// A value folded out of a node graph: up to four components, and how many of
/// them mean anything.
///
/// Kept as bare floats rather than as mx::Value all the way down because every
/// operator below has to broadcast a scalar against a colour, and doing that
/// through the variant would mean a type switch per operand per node. The mx
/// type is put back on at the end, where the shader inputs are read.
struct Folded
{
    float v[4]{ 0.0f, 0.0f, 0.0f, 0.0f };
    int n = 0;
    bool isColor = false;
    bool ok = false;
};

Folded foldScalar(float x)
{
    Folded f;
    f.v[0] = x;
    f.n = 1;
    f.ok = true;
    return f;
}

/// Component i, with a one-component value broadcast across all of them --
/// which is what makes multiply(color3, float) mean what a shading artist
/// expects it to.
float component(const Folded& f, int i)
{
    if (!f.ok)
    {
        return 0.0f;
    }
    if (f.n <= 1)
    {
        return f.v[0];
    }
    return i < f.n ? f.v[i] : 0.0f;
}

int widthOfType(const std::string& type)
{
    if (type == "float" || type == "integer" || type == "boolean")
        return 1;
    if (type == "vector2")
        return 2;
    if (type == "color3" || type == "vector3")
        return 3;
    if (type == "color4" || type == "vector4")
        return 4;
    return 0;
}

Folded fromValue(const mx::ValuePtr& v)
{
    Folded f;
    if (!v)
    {
        return f;
    }
    if (v->isA<float>())
        return foldScalar(v->asA<float>());
    if (v->isA<int>())
        return foldScalar(static_cast<float>(v->asA<int>()));
    if (v->isA<bool>())
        return foldScalar(v->asA<bool>() ? 1.0f : 0.0f);
    if (v->isA<mx::Vector2>())
    {
        const mx::Vector2 a = v->asA<mx::Vector2>();
        f = { { a[0], a[1], 0.0f, 0.0f }, 2, false, true };
        return f;
    }
    if (v->isA<mx::Color3>())
    {
        const mx::Color3 a = v->asA<mx::Color3>();
        f = { { a[0], a[1], a[2], 0.0f }, 3, true, true };
        return f;
    }
    if (v->isA<mx::Vector3>())
    {
        const mx::Vector3 a = v->asA<mx::Vector3>();
        f = { { a[0], a[1], a[2], 0.0f }, 3, false, true };
        return f;
    }
    if (v->isA<mx::Color4>())
    {
        const mx::Color4 a = v->asA<mx::Color4>();
        f = { { a[0], a[1], a[2], a[3] }, 4, true, true };
        return f;
    }
    if (v->isA<mx::Vector4>())
    {
        const mx::Vector4 a = v->asA<mx::Vector4>();
        f = { { a[0], a[1], a[2], a[3] }, 4, false, true };
        return f;
    }
    return f;
}

/// The node an input is driven by, whether it is wired straight to it or
/// through a nodegraph's output.
mx::NodePtr upstreamNode(const mx::InputPtr& input)
{
    if (!input)
    {
        return nullptr;
    }
    if (const mx::OutputPtr connected = input->getConnectedOutput())
    {
        if (const mx::NodePtr n = connected->getConnectedNode())
        {
            return n;
        }
    }
    return input->getConnectedNode();
}

Folded foldNode(const mx::NodePtr& node, int depth);

/// An operand: its literal value, or the graph behind it, or `fallback` when the
/// node leaves it at its nodedef default and MaterialX has not filled one in.
Folded foldOperand(const mx::NodePtr& node, const char* name, int depth, Folded fallback)
{
    if (!node)
    {
        return fallback;
    }
    const mx::InputPtr input = node->getInput(name);
    if (!input)
    {
        return fallback;
    }
    if (input->hasValue())
    {
        const Folded f = fromValue(input->getValue());
        return f.ok ? f : fallback;
    }
    const Folded f = foldNode(upstreamNode(input), depth + 1);
    return f.ok ? f : Folded{};
}

Folded foldBinary(const mx::NodePtr& node, int depth, const char* aName, const char* bName,
                  float (*op)(float, float), Folded bFallback)
{
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    const Folded a = foldOperand(node, aName, depth, Folded{});
    if (!a.ok)
    {
        return Folded{};
    }
    const Folded b = foldOperand(node, bName, depth, bFallback);
    if (!b.ok)
    {
        return Folded{};
    }
    Folded r;
    r.n = std::max(a.n, b.n);
    r.isColor = a.isColor || b.isColor;
    r.ok = true;
    for (int i = 0; i < r.n; ++i)
    {
        r.v[i] = op(component(a, i), component(b, i));
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    return r;
}

/// Fold a subgraph whose leaves are all constants.
///
/// Everything here is arithmetic on values the document states outright, which
/// is the whole difference between this and a shader generator: no texture may
/// appear anywhere in the subgraph, because a texture's value is not known until
/// the pixel is shaded and this result becomes a single number in OpenPBRParams.
/// An operand that does not fold makes the whole node not fold, and the input
/// ends up in `unsupported` with the category named -- silently substituting a
/// default for half an expression would be worse than saying so.
Folded foldNode(const mx::NodePtr& node, int depth)
{
    Folded bad;
    // Graphs are acyclic by construction, but a hand-edited document need not be
    // and this walk has no other way to notice.
    if (!node || depth > 32)
    {
        return bad;
    }

    const std::string category = node->getCategory();
    const int outWidth = widthOfType(node->getType());
    const bool outIsColor = node->getType() == "color3" || node->getType() == "color4";

    const auto retype = [&](Folded f) {
        if (f.ok && outWidth > 0)
        {
            // A scalar driving a colour output is the broadcast the type system
            // is asking for; anything else keeps the width it folded to.
            if (f.n == 1 && outWidth > 1)
            {
                for (int i = 1; i < outWidth; ++i)
                {
                    f.v[i] = f.v[0];
                }
                f.n = outWidth;
            }
            f.isColor = outIsColor;
        }
        return f;
    };

    const auto unary = [&](float (*op)(float)) {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        if (!a.ok)
        {
            return Folded{};
        }
        Folded r = a;
        for (int i = 0; i < r.n; ++i)
        {
            r.v[i] = op(a.v[i]);
        }
        return r;
    };

    if (category == "constant")
    {
        return retype(foldOperand(node, "value", depth, Folded{}));
    }
    if (category == "dot")
    {
        return foldOperand(node, "in", depth, Folded{});
    }
    if (category == "add")
        return retype(foldBinary(node, depth, "in1", "in2", [](float a, float b) { return a + b; }, foldScalar(0.0f)));
    if (category == "subtract")
        return retype(foldBinary(node, depth, "in1", "in2", [](float a, float b) { return a - b; }, foldScalar(0.0f)));
    if (category == "multiply")
        return retype(foldBinary(node, depth, "in1", "in2", [](float a, float b) { return a * b; }, foldScalar(1.0f)));
    if (category == "divide")
        return retype(foldBinary(node, depth, "in1", "in2", [](float a, float b) { return b != 0.0f ? a / b : 0.0f; }, foldScalar(1.0f)));
    if (category == "modulo")
        return retype(foldBinary(
            node, depth, "in1", "in2", [](float a, float b) { return b != 0.0f ? std::fmod(a, b) : 0.0f; }, foldScalar(1.0f)));
    if (category == "power")
        return retype(foldBinary(node, depth, "in1", "in2", [](float a, float b) { return std::pow(a, b); }, foldScalar(1.0f)));
    if (category == "min")
        return retype(foldBinary(node, depth, "in1", "in2", [](float a, float b) { return std::min(a, b); }, foldScalar(0.0f)));
    if (category == "max")
        return retype(foldBinary(node, depth, "in1", "in2", [](float a, float b) { return std::max(a, b); }, foldScalar(0.0f)));
    if (category == "absval")
        return unary([](float a) { return std::fabs(a); });
    if (category == "floor")
        return unary([](float a) { return std::floor(a); });
    if (category == "ceil")
        return unary([](float a) { return std::ceil(a); });
    if (category == "round")
        return unary([](float a) { return std::round(a); });
    if (category == "sign")
        return unary([](float a) { return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f); });
    if (category == "sqrt")
        return unary([](float a) { return a > 0.0f ? std::sqrt(a) : 0.0f; });
    if (category == "clamp")
    {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        const Folded lo = foldOperand(node, "low", depth, foldScalar(0.0f));
        const Folded hi = foldOperand(node, "high", depth, foldScalar(1.0f));
        if (!a.ok || !lo.ok || !hi.ok)
        {
            return bad;
        }
        Folded r = a;
        for (int i = 0; i < r.n; ++i)
        {
            r.v[i] = std::clamp(component(a, i), component(lo, i), component(hi, i));
        }
        return r;
    }
    if (category == "mix")
    {
        // mix = 0 is bg and mix = 1 is fg, which is the way round the stdlib
        // defines it and the opposite of the argument order most libraries use.
        const Folded fg = foldOperand(node, "fg", depth, Folded{});
        const Folded bg = foldOperand(node, "bg", depth, Folded{});
        const Folded t = foldOperand(node, "mix", depth, foldScalar(0.0f));
        if (!fg.ok || !bg.ok || !t.ok)
        {
            return bad;
        }
        Folded r;
        r.n = std::max(fg.n, bg.n);
        r.isColor = fg.isColor || bg.isColor;
        r.ok = true;
        for (int i = 0; i < r.n; ++i)
        {
            const float k = component(t, i);
            r.v[i] = component(bg, i) * (1.0f - k) + component(fg, i) * k;
        }
        return r;
    }
    if (category == "remap")
    {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        const Folded il = foldOperand(node, "inlow", depth, foldScalar(0.0f));
        const Folded ih = foldOperand(node, "inhigh", depth, foldScalar(1.0f));
        const Folded ol = foldOperand(node, "outlow", depth, foldScalar(0.0f));
        const Folded oh = foldOperand(node, "outhigh", depth, foldScalar(1.0f));
        if (!a.ok || !il.ok || !ih.ok || !ol.ok || !oh.ok)
        {
            return bad;
        }
        Folded r = a;
        for (int i = 0; i < r.n; ++i)
        {
            const float span = component(ih, i) - component(il, i);
            const float u = span != 0.0f ? (component(a, i) - component(il, i)) / span : 0.0f;
            r.v[i] = component(ol, i) + u * (component(oh, i) - component(ol, i));
        }
        return r;
    }
    if (category == "smoothstep")
    {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        const Folded lo = foldOperand(node, "low", depth, foldScalar(0.0f));
        const Folded hi = foldOperand(node, "high", depth, foldScalar(1.0f));
        if (!a.ok || !lo.ok || !hi.ok)
        {
            return bad;
        }
        Folded r = a;
        for (int i = 0; i < r.n; ++i)
        {
            const float span = component(hi, i) - component(lo, i);
            const float u = span != 0.0f ? std::clamp((component(a, i) - component(lo, i)) / span, 0.0f, 1.0f) : 0.0f;
            r.v[i] = u * u * (3.0f - 2.0f * u);
        }
        return r;
    }
    if (category == "invert")
    {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        const Folded amount = foldOperand(node, "amount", depth, foldScalar(1.0f));
        if (!a.ok || !amount.ok)
        {
            return bad;
        }
        Folded r = a;
        r.n = std::max(a.n, amount.n);
        for (int i = 0; i < r.n; ++i)
        {
            r.v[i] = component(amount, i) - component(a, i);
        }
        return r;
    }
    if (category == "normalize" || category == "magnitude" || category == "dotproduct")
    {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        if (category == "dotproduct")
        {
            const Folded b = foldOperand(node, "in2", depth, Folded{});
            const Folded a2 = a.ok ? a : foldOperand(node, "in1", depth, Folded{});
            if (!a2.ok || !b.ok)
            {
                return bad;
            }
            float d = 0.0f;
            for (int i = 0; i < std::max(a2.n, b.n); ++i)
            {
                d += component(a2, i) * component(b, i);
            }
            return foldScalar(d);
        }
        if (!a.ok)
        {
            return bad;
        }
        float len2 = 0.0f;
        for (int i = 0; i < a.n; ++i)
        {
            len2 += a.v[i] * a.v[i];
        }
        const float len = std::sqrt(len2);
        if (category == "magnitude")
        {
            return foldScalar(len);
        }
        Folded r = a;
        for (int i = 0; i < r.n; ++i)
        {
            r.v[i] = len > 0.0f ? a.v[i] / len : 0.0f;
        }
        return r;
    }
    if (category == "luminance")
    {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        // The stdlib default lumacoeffs, which are ACEScg's and not Rec.709's --
        // taking the familiar ones would be a quiet 3% on every grey.
        const Folded c =
            foldOperand(node, "lumacoeffs", depth, Folded{ { 0.272229f, 0.674082f, 0.053690f, 0.0f }, 3, false, true });
        if (!a.ok || !c.ok)
        {
            return bad;
        }
        float l = 0.0f;
        for (int i = 0; i < 3; ++i)
        {
            l += component(a, i) * component(c, i);
        }
        Folded r = a;
        for (int i = 0; i < r.n && i < 3; ++i)
        {
            r.v[i] = l;
        }
        return r;
    }
    if (category == "combine2" || category == "combine3" || category == "combine4")
    {
        const int count = category == "combine2" ? 2 : (category == "combine3" ? 3 : 4);
        Folded r;
        r.n = count;
        r.isColor = outIsColor;
        r.ok = true;
        const char* const names[4] = { "in1", "in2", "in3", "in4" };
        for (int i = 0; i < count; ++i)
        {
            const Folded a = foldOperand(node, names[i], depth, foldScalar(0.0f));
            if (!a.ok)
            {
                return bad;
            }
            r.v[i] = a.v[0];
        }
        return r;
    }
    if (category == "extract")
    {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        const Folded idx = foldOperand(node, "index", depth, foldScalar(0.0f));
        if (!a.ok || !idx.ok)
        {
            return bad;
        }
        const int i = std::clamp(static_cast<int>(idx.v[0]), 0, 3);
        return foldScalar(a.n > i ? a.v[i] : 0.0f);
    }
    if (category == "swizzle")
    {
        const Folded a = foldOperand(node, "in", depth, Folded{});
        const mx::InputPtr channels = node->getInput("channels");
        if (!a.ok || !channels)
        {
            return bad;
        }
        const std::string spec = channels->getValueString();
        Folded r;
        r.isColor = outIsColor;
        r.ok = true;
        for (const char ch : spec)
        {
            if (r.n >= 4)
            {
                break;
            }
            float value = 0.0f;
            switch (ch)
            {
            case 'r':
            case 'x':
                value = component(a, 0);
                break;
            case 'g':
            case 'y':
                value = component(a, 1);
                break;
            case 'b':
            case 'z':
                value = component(a, 2);
                break;
            case 'a':
            case 'w':
                value = component(a, 3);
                break;
            case '0':
                value = 0.0f;
                break;
            case '1':
                value = 1.0f;
                break;
            default:
                return bad;
            }
            r.v[r.n++] = value;
        }
        return r.n > 0 ? r : bad;
    }
    if (category == "convert")
    {
        return retype(foldOperand(node, "in", depth, Folded{}));
    }
    if (category == "ifgreater" || category == "ifgreatereq" || category == "ifequal")
    {
        const Folded v1 = foldOperand(node, "value1", depth, Folded{});
        const Folded v2 = foldOperand(node, "value2", depth, Folded{});
        const Folded a = foldOperand(node, "in1", depth, Folded{});
        const Folded b = foldOperand(node, "in2", depth, Folded{});
        if (!v1.ok || !v2.ok || !a.ok || !b.ok)
        {
            return bad;
        }
        const bool taken = category == "ifgreater"     ? (v1.v[0] > v2.v[0]) :
                           (category == "ifgreatereq") ? (v1.v[0] >= v2.v[0]) :
                                                         (v1.v[0] == v2.v[0]);
        return taken ? a : b;
    }

    return bad;
}

/// Put the mx type back on, so the shader-input readers stay unchanged.
mx::ValuePtr foldedToValue(const Folded& f)
{
    switch (f.n)
    {
    case 1:
        return mx::Value::createValue<float>(f.v[0]);
    case 2:
        return mx::Value::createValue<mx::Vector2>(mx::Vector2(f.v[0], f.v[1]));
    case 3:
        return f.isColor ? mx::Value::createValue<mx::Color3>(mx::Color3(f.v[0], f.v[1], f.v[2])) :
                           mx::Value::createValue<mx::Vector3>(mx::Vector3(f.v[0], f.v[1], f.v[2]));
    case 4:
        return f.isColor ? mx::Value::createValue<mx::Color4>(mx::Color4(f.v[0], f.v[1], f.v[2], f.v[3])) :
                           mx::Value::createValue<mx::Vector4>(mx::Vector4(f.v[0], f.v[1], f.v[2], f.v[3]));
    default:
        return nullptr;
    }
}

/// Read the placement of an image node: <tiledimage>'s own tiling, and whatever
/// <place2d> feeds its texcoord.
///
/// Both are transcribed from the stdlib nodegraphs rather than from the field
/// names, because the names mislead in both nodes. place2d *divides* by scale
/// and *subtracts* offset, around a pivot; tiledimage subtracts its offset
/// after multiplying. And MaterialX's rotate2d is (ca*x + sa*y, -sa*x + ca*y),
/// which is a clockwise rotation, while the shader's is counter-clockwise --
/// hence the negated angle. Getting any of these backwards produces a texture
/// that is placed *almost* right, which is the hardest kind of wrong to see.
UvPlacement readPlacement(const mx::NodePtr& node, std::string& gap)
{
    UvPlacement out;
    if (!node)
    {
        return out;
    }

    if (node->getCategory() == "tiledimage")
    {
        const Folded tiling = foldOperand(node, "uvtiling", 0, Folded{ { 1.0f, 1.0f, 0.0f, 0.0f }, 2, false, true });
        const Folded offset = foldOperand(node, "uvoffset", 0, Folded{ { 0.0f, 0.0f, 0.0f, 0.0f }, 2, false, true });
        if (tiling.ok && offset.ok)
        {
            out.scaleX = component(tiling, 0);
            out.scaleY = component(tiling, 1);
            out.offsetX = -component(offset, 0);
            out.offsetY = -component(offset, 1);
            out.stated = out.scaleX != 1.0f || out.scaleY != 1.0f || out.offsetX != 0.0f || out.offsetY != 0.0f;
        }
        // Real-world sizing needs the scene's unit system, which the renderer
        // does not carry. Named rather than silently applied as a factor of one.
        for (const char* input : { "realworldimagesize", "realworldtilesize" })
        {
            if (const mx::InputPtr i = node->getInput(input); i && (i->hasValue() || i->getConnectedNode()))
            {
                gap = input;
            }
        }
    }

    const mx::InputPtr texcoord = node->getInput("texcoord");
    const mx::NodePtr placer = upstreamNode(texcoord);
    if (!placer)
    {
        return out;
    }
    const std::string placerCategory = placer->getCategory();
    // A bare <texcoord> is the identity, and the mesh's own UVs are what the
    // shader already samples with.
    if (placerCategory == "texcoord")
    {
        return out;
    }
    if (placerCategory != "place2d")
    {
        gap = placerCategory;
        return out;
    }
    const mx::InputPtr order = placer->getInput("operationorder");
    const Folded orderValue = order && order->hasValue() ? fromValue(order->getValue()) : foldScalar(0.0f);
    if (orderValue.ok && orderValue.v[0] != 0.0f)
    {
        // The other order applies translation before scale and rotation, which
        // is a different transform and not the one composed below.
        gap = "place2d(operationorder)";
        return out;
    }

    const Folded pivot = foldOperand(placer, "pivot", 0, Folded{ { 0.0f, 0.0f, 0.0f, 0.0f }, 2, false, true });
    const Folded scale = foldOperand(placer, "scale", 0, Folded{ { 1.0f, 1.0f, 0.0f, 0.0f }, 2, false, true });
    const Folded rotate = foldOperand(placer, "rotate", 0, foldScalar(0.0f));
    const Folded offset = foldOperand(placer, "offset", 0, Folded{ { 0.0f, 0.0f, 0.0f, 0.0f }, 2, false, true });
    if (!pivot.ok || !scale.ok || !rotate.ok || !offset.ok)
    {
        gap = "place2d(non-constant)";
        return out;
    }

    const float sx = component(scale, 0);
    const float sy = component(scale, 1);
    if (sx == 0.0f || sy == 0.0f)
    {
        gap = "place2d(zero scale)";
        return out;
    }

    // out = R((uv - pivot)/scale) - offset + pivot, so the shader's
    // rotate(uv * k) + off matches with k = 1/scale and the pivot terms folded
    // into off.
    const float theta = -static_cast<float>(rotate.v[0] * M_PI / 180.0);
    const float c = std::cos(theta);
    const float sn = std::sin(theta);
    const float px = component(pivot, 0) / sx;
    const float py = component(pivot, 1) / sy;
    const float rotatedPivotX = px * c - py * sn;
    const float rotatedPivotY = px * sn + py * c;

    UvPlacement fromPlacer;
    fromPlacer.scaleX = 1.0f / sx;
    fromPlacer.scaleY = 1.0f / sy;
    fromPlacer.rotation = theta;
    fromPlacer.offsetX = component(pivot, 0) - component(offset, 0) - rotatedPivotX;
    fromPlacer.offsetY = component(pivot, 1) - component(offset, 1) - rotatedPivotY;
    fromPlacer.stated = true;

    if (out.stated)
    {
        // A tiled image *and* a placement in front of it compose into a
        // transform this single scale/rotate/offset cannot always express --
        // a per-axis scale after a rotation is not of that form.
        gap = "place2d+tiledimage";
        return out;
    }
    return fromPlacer;
}

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
            // Only what this element states, never what it inherits. A document
            // declares its *working* space on the root element, and reading
            // that as the file's would relabel every untagged map -- the chess
            // set tags its base colours srgb_texture and leaves metalness,
            // roughness and normals bare, which is exactly the arrangement the
            // renderer's own slot defaults already assume.
            out.colorSpace = file->getAttribute("colorspace");
        }
        if (out.colorSpace.empty())
        {
            out.colorSpace = node->getAttribute("colorspace");
        }
        std::string placementGap;
        out.placement = readPlacement(node, placementGap);
        if (!placementGap.empty())
        {
            out.unsupportedCategory = placementGap;
        }
        return out;
    }

    // Not an image, so the last chance is that the graph behind it is
    // arithmetic on constants -- a tint scaled, two colours mixed, a roughness
    // remapped. Those are what a real library is made of, and until this fold
    // existed every one of them landed in `unsupported` and left the parameter
    // at its specification default: a material that quietly ignored what the
    // document said, while reporting success.
    if (const Folded folded = foldNode(node, 0); folded.ok)
    {
        out.isConstant = true;
        out.value = foldedToValue(folded);
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

/// The document's colorspace name, reduced to the one question the renderer can
/// answer: is this file gamma-encoded or is it linear?
///
/// Matched on the naming convention rather than a table of every name, because
/// the set is open -- a studio config adds its own -- and the convention is what
/// the names are built from: `lin_` and the ACES spaces are linear, `srgb`,
/// `g22`, `g18` and the display spaces carry a curve. What is *not* answered is
/// the gamut: acescg and lin_rec709 both come back Linear even though their
/// primaries differ, because there is no colour management here to convert them
/// with. A name that fits neither pattern is reported and left to the slot.
TexColorSpace colorSpaceFromName(const std::string& raw, const std::string& forInput)
{
    if (raw.empty())
    {
        return TexColorSpace::Unspecified;
    }
    std::string name;
    name.reserve(raw.size());
    for (const char c : raw)
    {
        name.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    const auto startsWith = [&name](const char* prefix) { return name.starts_with(prefix); };

    if (startsWith("lin_") || startsWith("linear") || name == "acescg" || name == "acescc" || name == "aces2065-1" ||
        name == "raw" || name == "none" || name == "scene-linear")
    {
        return TexColorSpace::Linear;
    }
    if (startsWith("srgb") || startsWith("g22") || startsWith("g18") || startsWith("gamma") || name == "adobergb" ||
        name == "rec709_display" || name == "rec709")
    {
        return TexColorSpace::Srgb;
    }
    STRELKA_WARNING(
        "MaterialX: colorspace '{}' on '{}' is not one this renderer can decode; "
        "using the slot's own default. Only the transfer function is honoured, never the gamut.",
        raw, forInput);
    return TexColorSpace::Unspecified;
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

    // Absolute, because a texture path is later joined with the *scene's*
    // directory: a relative docDir would prefix it a second time and every map
    // would fail to open. The scene path arrives from the CLI, so it is
    // routinely relative.
    const mx::FilePath docDir(std::filesystem::absolute(path).parent_path().string());

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
        // OpenPBRParams carries one UV transform for the whole material, so the
        // first placement stated wins and a second, different one is reported.
        UvPlacement placement;
        bool sawPlacement = false;

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
                out.texColorSpace[(size_t)binding.textureSlot] = colorSpaceFromName(r.colorSpace, name);
                if (r.placement.stated)
                {
                    if (!sawPlacement)
                    {
                        placement = r.placement;
                        sawPlacement = true;
                    }
                    else if (!placementsAgree(placement, r.placement))
                    {
                        out.unsupported.push_back(name + "<-place2d(second placement)");
                    }
                }
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

        if (sawPlacement)
        {
            out.params.uv_scale_x = placement.scaleX;
            out.params.uv_scale_y = placement.scaleY;
            out.params.uv_rotation = placement.rotation;
            out.params.uv_offset_x = placement.offsetX;
            out.params.uv_offset_y = placement.offsetY;
        }

        // A mapped subsurface weight leaves the constant at zero, and the
        // constant is not only a default: the wavefront tracer's `extend` stage
        // has a medium id and no UV, so it derives the interior volume from this
        // block alone. At weight zero that volume is *no medium* -- zero
        // extinction -- and the random walk then crosses the object in one
        // straight line and leaves at full throughput, which is what rendered
        // the Open Chess Set's kings white instead of dark marble. Where the map
        // reads zero no walk starts in the first place, so saying "there is a
        // medium here" costs nothing and is what the document means.
        if (!out.texPaths[OPENPBR_TEX_SUBSURFACE_WEIGHT].empty() && out.params.subsurface_weight <= 0.0f)
        {
            out.params.subsurface_weight = 1.0f;
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
            desc.openpbrTexColorSpace = m.texColorSpace;
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
        desc.openpbrTexColorSpace = found->texColorSpace;
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
