#include "MdlDiscoveryPlugin.h"

#include <pxr/base/tf/staticTokens.h>

//#include "Tokens.h"

PXR_NAMESPACE_OPEN_SCOPE
// clang-format off
TF_DEFINE_PRIVATE_TOKENS(_tokens,
        (mdl)
);
// clang-format on

NDR_REGISTER_DISCOVERY_PLUGIN(HdStrelkaMdlDiscoveryPlugin);

NdrNodeDiscoveryResultVec HdStrelkaMdlDiscoveryPlugin::DiscoverNodes(const Context& ctx)
{
    NdrNodeDiscoveryResultVec result;

    NdrNodeDiscoveryResult mdlNode(
        /* identifier    */ _tokens->mdl,
        /* version       */ NdrVersion(1),
        /* name          */ _tokens->mdl,
        /* family        */ TfToken(),
        /* discoveryType */ _tokens->mdl,
        /* sourceType    */ _tokens->mdl,
        /* uri           */ std::string(),
        /* resolvedUri   */ std::string());
    result.push_back(mdlNode);

    return result;
}

const NdrStringVec& HdStrelkaMdlDiscoveryPlugin::GetSearchURIs() const
{
  static const NdrStringVec s_searchURIs;
  return s_searchURIs;
}

PXR_NAMESPACE_CLOSE_SCOPE
