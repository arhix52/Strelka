#include "bvh.h"

#include <algorithm>

namespace oka
{
BvhBuilder::BvhBuilder()
{
    mDevice = rtcNewDevice(nullptr);
}

BvhBuilder::~BvhBuilder()
{
    rtcReleaseDevice(mDevice);
}

BvhBuilder::AABB BvhBuilder::computeBounds(const std::vector<BvhNodeInternal>& nodes, uint32_t start, uint32_t end)
{
    AABB result;
    for (uint32_t i = start; i < end; ++i)
    {
        const BvhNodeInternal& n = nodes[i];
        result.expand(n.box);
    }
    return result;
}

BvhBuilder::AABB BvhBuilder::computeCentroids(const std::vector<BvhNodeInternal>& nodes, uint32_t start, uint32_t end)
{
    AABB result;
    for (uint32_t i = start; i < end; ++i)
    {
        const BvhNodeInternal& n = nodes[i];
        result.expand(n.box.getCentroid());
    }
    return result;
}

void BvhBuilder::setDepthFirstVisitOrder(std::vector<BvhNodeInternal>& nodes,
                                         uint32_t nodeId,
                                         uint32_t nextId,
                                         uint32_t& order)
{
    BvhNodeInternal& node = nodes[nodeId];
    node.visitOrder = order++;
    node.next = nextId;
    if (node.left != InvalidMask)
    {
        setDepthFirstVisitOrder(nodes, node.left, node.right, order);
    }
    if (node.right != InvalidMask)
    {
        setDepthFirstVisitOrder(nodes, node.right, nextId, order);
    }
}

void BvhBuilder::setDepthFirstVisitOrder(std::vector<BvhNodeInternal>& nodes, uint32_t root)
{
    uint32_t order = 0;
    setDepthFirstVisitOrder(nodes, root, InvalidMask, order);
}

/* This function is called by the builder to signal progress and to
 * report memory consumption. */
bool memoryMonitor(void* userPtr, ssize_t bytes, bool post)
{
    (void)userPtr;
    (void)bytes;
    (void)post;
    return true;
}

bool buildProgress(void* userPtr, double f)
{
    (void)userPtr;
    (void)f;
    return true;
}

void BvhBuilder::splitPrimitive(
    const RTCBuildPrimitive* prim, unsigned int dim, float pos, RTCBounds* lprim, RTCBounds* rprim, void* userPtr)
{
    (void)userPtr;
    assert(dim < 3);
    // assert(prim->geomID == 0);

    lprim->lower_x = prim->lower_x;
    lprim->lower_y = prim->lower_y;
    lprim->lower_z = prim->lower_z;

    lprim->upper_x = prim->upper_x;
    lprim->upper_y = prim->upper_y;
    lprim->upper_z = prim->upper_z;

    rprim->lower_x = prim->lower_x;
    rprim->lower_y = prim->lower_y;
    rprim->lower_z = prim->lower_z;

    rprim->upper_x = prim->upper_x;
    rprim->upper_y = prim->upper_y;
    rprim->upper_z = prim->upper_z;

    (&lprim->upper_x)[dim] = pos;
    (&rprim->lower_x)[dim] = pos;
}

BVH BvhBuilder::buildEmbree(const std::vector<BVHInputPosition>& positions)
{
    const uint32_t totalTriangles = (uint32_t)positions.size() / 3;
    std::vector<RTCBuildPrimitive> prims;
    const size_t extraSpace = totalTriangles / 2; // TODO: check this parameter
    prims.reserve(totalTriangles + extraSpace);
    prims.resize(totalTriangles);
    for (uint32_t i = 0; i < totalTriangles; ++i)
    {
        RTCBuildPrimitive& prim = prims[i];
        Triangle triangle{};
        triangle.v0 = positions[i * 3 + 0].pos;
        triangle.v1 = positions[i * 3 + 1].pos;
        triangle.v2 = positions[i * 3 + 2].pos;
        AABB box = boundingBox(triangle);

        prim.geomID = positions[i * 3 + 0].instId;
        prim.primID = positions[i * 3 + 0].primId;
        prim.lower_x = box.minimum.x;
        prim.lower_y = box.minimum.y;
        prim.lower_z = box.minimum.z;
        prim.upper_x = box.maximum.x;
        prim.upper_y = box.maximum.y;
        prim.upper_z = box.maximum.z;
    }

    RTCBVH embreeBvh = rtcNewBVH(mDevice);

    RTCBuildArguments arguments = rtcDefaultBuildArguments();
    arguments.byteSize = sizeof(arguments);
    arguments.buildFlags = RTC_BUILD_FLAG_NONE;
    arguments.buildQuality = RTC_BUILD_QUALITY_HIGH;
    arguments.maxBranchingFactor = 2;
    arguments.maxDepth = 1024;
    arguments.sahBlockSize = 1;
    arguments.minLeafSize = 1;
    arguments.maxLeafSize = 1;
    arguments.traversalCost = 1.0f;
    arguments.intersectionCost = 1.0f;
    arguments.bvh = embreeBvh;
    arguments.primitives = prims.data();
    arguments.primitiveCount = totalTriangles; // prims.size();
    arguments.primitiveArrayCapacity = prims.capacity();
    arguments.createNode = InnerNode::create;
    arguments.setNodeChildren = InnerNode::setChildren;
    arguments.setNodeBounds = InnerNode::setBounds;
    arguments.createLeaf = LeafNode::create;
    arguments.splitPrimitive = BvhBuilder::splitPrimitive;
    arguments.buildProgress = nullptr;
    arguments.userPtr = nullptr;

    Node* root = (Node*)rtcBuildBVH(&arguments);
    uint32_t order = 0;
    setDepthFirstVisitOrder(root, order);

    BVH res = repackEmbree(root, positions, order, totalTriangles);

    rtcReleaseBVH(embreeBvh);

    return res;
}

void BvhBuilder::setDepthFirstVisitOrder(Node* current, uint32_t& order)
{
    current->visitOrder = order++;
    if (current->children[0])
    {
        setDepthFirstVisitOrder(current->children[0], order);
    }
    if (current->children[1])
    {
        setDepthFirstVisitOrder(current->children[1], order);
    }
}

void BvhBuilder::repackEmbree(const Node* current,
                              const std::vector<BVHInputPosition>& positions,
                              BVH& outBvh,
                              uint32_t& positionInArray,
                              uint32_t& positionInTrianglesArray,
                              const uint32_t nextId)
{
    outBvh.nodes.push_back({});
    BVHNode& curr = outBvh.nodes[positionInArray++];
    curr.nodeOffset = nextId;
    if (!current->children[0] && !current->children[1])
    {
        // leaf
        union
        {
            uint32_t primUint;
            float primFloat;
        };
        primUint = ((LeafNode*)current)->mTriangleId;
        curr.instId = ((LeafNode*)current)->mInstId;
        curr.minBounds.x = primFloat;
    }
    else
    {
        // inner
        curr.minBounds = current->bounds.minimum;
        curr.maxBounds = current->bounds.maximum;
        curr.instId = InvalidMask;
        if (current->children[0])
        {
            uint32_t offset = current->children[1] ? current->children[1]->visitOrder : InvalidMask;
            repackEmbree(current->children[0], positions, outBvh, positionInArray, positionInTrianglesArray, offset);
        }
        if (current->children[1])
        {
            repackEmbree(current->children[1], positions, outBvh, positionInArray, positionInTrianglesArray, nextId);
        }
    }
}

BVH BvhBuilder::repackEmbree(const Node* root,
                             const std::vector<BVHInputPosition>& positions,
                             const uint32_t totalNodes,
                             const uint32_t totalTriangles)
{
    (void)totalTriangles;
    BVH res;
    res.nodes.reserve(totalNodes);
    uint32_t posInNodeArray = 0;
    uint32_t posInTriangleArray = 0;
    repackEmbree(root, positions, res, posInNodeArray, posInTriangleArray, InvalidMask);

    return res;
}

BVH BvhBuilder::build(const std::vector<BVHInputPosition>& positions)
{
    const uint32_t totalTriangles = (uint32_t)positions.size() / 3;
    if (totalTriangles == 0)
    {
        return BVH();
    }

    if (mUseEmbree)
    {
        return buildEmbree(positions);
    }
    else
    {
        assert(0); // notsupported;
        return BVH();
    }
}

BVH BvhBuilder::repack(const std::vector<BvhNodeInternal>& nodes, const uint32_t totalTriangles)
{
    (void)totalTriangles;
    BVH res;
    res.nodes.resize(nodes.size());
    uint32_t triangleId = 0;
    for (uint32_t i = 0; i < (uint32_t)nodes.size(); ++i)
    {
        BvhNodeInternal const& oldNode = nodes[i];
        BVHNode& newNode = res.nodes[oldNode.visitOrder];
        newNode.nodeOffset = oldNode.next == InvalidMask ? InvalidMask : nodes[oldNode.next].visitOrder;
        newNode.instId = oldNode.prim;
        if (oldNode.prim != InvalidMask) // leaf
        {
            newNode.minBounds = oldNode.triangle.v1 - oldNode.triangle.v0;
            newNode.maxBounds = oldNode.triangle.v2 - oldNode.triangle.v0;
            newNode.instId = triangleId;
            ++triangleId;
        }
        else
        {
            newNode.minBounds = oldNode.box.minimum;
            newNode.maxBounds = oldNode.box.maximum;
        }
    }
    return res;
}

} // namespace oka
