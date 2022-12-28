#pragma once
#define GLM_FORCE_SILENT_WARNINGS
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <embree3/rtcore.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/compatibility.hpp>
RTC_NAMESPACE_USE
#include <vector>


namespace oka
{

// GPU structure
struct BVHNode
{
    glm::float3 minBounds = { 0.0f, 0.0f, 0.0f };
    int instId = (int)0xFFFFFFFF;
    glm::float3 maxBounds = { 0.0f, 0.0f, 0.0f };
    int nodeOffset = (int)0xFFFFFFFF;
};

struct BVH
{
    std::vector<BVHNode> nodes;
};

struct BVHInputPosition
{
    glm::float3 pos;
    uint32_t instId = 0xFFFFFFFF;
    uint32_t primId = 0xFFFFFFFF;
};

class BvhBuilder
{
public:
    bool mUseEmbree = true;

    BvhBuilder();
    ~BvhBuilder();

    BVH build(const std::vector<BVHInputPosition>& positions);

private:
    RTCDevice mDevice = nullptr;

    struct AABB
    {
        glm::float3 minimum{ 1e10f };
        glm::float3 maximum{ -1e10f };
        AABB()
        {
        }
        AABB(const glm::float3& a, const glm::float3& b)
        {
            minimum = a;
            maximum = b;
        }

        static float halfArea(const glm::float3& d)
        {
            return d.x * (d.y + d.z) + (d.y * d.z);
        }

        static float area(const AABB& a)
        {
            return 2.0f * halfArea(a.getSize());
        }

        static AABB Union(const AABB& a, const AABB& b)
        {
            AABB res;
            res.expand(a);
            res.expand(b);
            return res;
        }

        glm::float3 getCentroid() const
        {
            return (minimum + maximum) * 0.5f;
        }

        glm::float3 getSize() const
        {
            return maximum - minimum;
        }

        // returns number of axis
        uint32_t getMaxinumExtent()
        {
            uint32_t res = 2;
            glm::float3 dim = maximum - minimum;
            if (dim.x > dim.y && dim.x > dim.z)
            {
                res = 0;
            }
            else if (dim.y > dim.x && dim.y > dim.z)
            {
                res = 1;
            }
            return res;
        }

        void expand(const glm::float3& p)
        {
            minimum.x = std::min(minimum.x, p.x);
            minimum.y = std::min(minimum.y, p.y);
            minimum.z = std::min(minimum.z, p.z);

            maximum.x = std::max(maximum.x, p.x);
            maximum.y = std::max(maximum.y, p.y);
            maximum.z = std::max(maximum.z, p.z);
        }

        void expand(const AABB& o)
        {
            minimum.x = std::min(minimum.x, o.minimum.x);
            minimum.y = std::min(minimum.y, o.minimum.y);
            minimum.z = std::min(minimum.z, o.minimum.z);

            maximum.x = std::max(maximum.x, o.maximum.x);
            maximum.y = std::max(maximum.y, o.maximum.y);
            maximum.z = std::max(maximum.z, o.maximum.z);
        }
    };

    struct Triangle
    {
        glm::float3 v0;
        glm::float3 v1;
        glm::float3 v2;
    };

    AABB boundingBox(const Triangle& tri)
    {
        float minX = std::min(tri.v0.x, std::min(tri.v1.x, tri.v2.x));
        float minY = std::min(tri.v0.y, std::min(tri.v1.y, tri.v2.y));
        float minZ = std::min(tri.v0.z, std::min(tri.v1.z, tri.v2.z));

        float maxX = std::max(tri.v0.x, std::max(tri.v1.x, tri.v2.x));
        float maxY = std::max(tri.v0.y, std::max(tri.v1.y, tri.v2.y));
        float maxZ = std::max(tri.v0.z, std::max(tri.v1.z, tri.v2.z));

        const float eps = 1e-6f;
        // need to pad aabb to prevent from ultra thin box (zero width)
        return AABB(glm::float3(minX, minY, minZ), glm::float3(maxX + eps, maxY + eps, maxZ + eps));
    }

    static const uint32_t LeafMask = 0x80000000;
    static const uint32_t InvalidMask = 0xFFFFFFFF;

    struct Node
    {
        AABB bounds;
        Node* children[2] = { nullptr, nullptr };
        uint32_t visitOrder = InvalidMask;
        virtual float sah() = 0;
    };

    struct InnerNode : public Node
    {
        InnerNode()
        {
            children[0] = children[1] = nullptr;
        }

        float sah()
        {
            const AABB& leftChild = children[0]->bounds;
            const AABB& rightChild = children[1]->bounds;
            return 1.0f + (AABB::area(leftChild) * children[0]->sah() + AABB::area(rightChild) * children[1]->sah()) /
                              AABB::area(AABB::Union(leftChild, rightChild));
        }

        static void* create(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr)
        {
            (void)userPtr;
            (void)numChildren;
            assert(numChildren == 2);
            void* ptr = rtcThreadLocalAlloc(alloc, sizeof(InnerNode), 16);
            return (void*)new (ptr) InnerNode;
        }

        static void setChildren(void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr)
        {
            (void)userPtr;
            (void)numChildren;
            assert(numChildren == 2);
            for (size_t i = 0; i < 2; i++)
            {
                ((InnerNode*)nodePtr)->children[i] = (Node*)childPtr[i];
            }
        }

        static void setBounds(void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr)
        {
            (void)userPtr;
            (void)numChildren;
            assert(numChildren == 2);
            const RTCBounds* leftChildBounds = bounds[0];
            const RTCBounds* rightChildBounds = bounds[1];
            AABB leftChildAABB(glm::float3(leftChildBounds->lower_x, leftChildBounds->lower_y, leftChildBounds->lower_z),
                               glm::float3(leftChildBounds->upper_x, leftChildBounds->upper_y, leftChildBounds->upper_z));
            AABB rightChildAABB(
                glm::float3(rightChildBounds->lower_x, rightChildBounds->lower_y, rightChildBounds->lower_z),
                glm::float3(rightChildBounds->upper_x, rightChildBounds->upper_y, rightChildBounds->upper_z));
            ((InnerNode*)nodePtr)->bounds = AABB::Union(leftChildAABB, rightChildAABB);
        }
    };

    struct LeafNode : public Node
    {
        unsigned mTriangleId;
        unsigned mInstId;

        LeafNode(unsigned triangleId, unsigned instId, const AABB& bounds) : mTriangleId(triangleId), mInstId(instId)
        {
            this->bounds = bounds;
        }

        float sah()
        {
            return 1.0f;
        }

        static void* create(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr)
        {
            (void)userPtr;
            (void)numPrims;
            assert(numPrims == 1);
            void* ptr = rtcThreadLocalAlloc(alloc, sizeof(LeafNode), 16);
            return (void*)new (ptr) LeafNode(prims->primID, prims->geomID, *(AABB*)prims);
        }
    };


    struct BvhNodeInternal
    {
        BvhNodeInternal(){};
        AABB box;
        bool isLeaf = false;
        uint32_t visitOrder = InvalidMask;
        uint32_t next = InvalidMask;
        uint32_t prim = InvalidMask;
        uint32_t left = InvalidMask;
        uint32_t right = InvalidMask;
        Triangle triangle; // only for leaf
    };

    void setDepthFirstVisitOrder(std::vector<BvhNodeInternal>& nodes, uint32_t nodeId, uint32_t nextId, uint32_t& order);
    void setDepthFirstVisitOrder(std::vector<BvhNodeInternal>& nodes, uint32_t root);

    AABB computeBounds(const std::vector<BvhNodeInternal>& nodes, uint32_t start, uint32_t end);
    AABB computeCentroids(const std::vector<BvhNodeInternal>& nodes, uint32_t start, uint32_t end);
    BVH repack(const std::vector<BvhNodeInternal>& nodes, const uint32_t totalTriangles);

    // embree
    BVH buildEmbree(const std::vector<BVHInputPosition>& positions);
    BVH repackEmbree(const Node* root,
                     const std::vector<BVHInputPosition>& positions,
                     const uint32_t totalNodes,
                     const uint32_t totalTriangles);
    void setDepthFirstVisitOrder(Node* current, uint32_t& order);
    void repackEmbree(const Node* current,
                      const std::vector<BVHInputPosition>& positions,
                      BVH& outBvh,
                      uint32_t& positionInArray,
                      uint32_t& positionInTrianglesArray,
                      const uint32_t nextId);
    static void splitPrimitive(
        const RTCBuildPrimitive* prim, unsigned int dim, float pos, RTCBounds* lprim, RTCBounds* rprim, void* userPtr);
};

} // namespace oka
