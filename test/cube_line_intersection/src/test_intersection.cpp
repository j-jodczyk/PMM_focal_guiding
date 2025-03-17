#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>

using Vector = Eigen::Vector3d;

struct Point {
    float x;
    float y;
    float z;

    Point() {}

    Point(float _x, float _y, float _z) {
        x = _x;
        y = _y;
        z = _z;
    }

    Vector operator-(const Point &p) const {
        Vector v(3);
        v << x - p.x, y - p.y, z - p.z;
        return v;
    }

    Point operator+(const Point &p) const {
        return Point(x + p.x, y + p.y, z + p.z);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Point[" << x << ", " << y << ", " << z << "]";
        return oss.str();
    }
};

struct AABB {
    AABB() {};

    AABB(Point _min, Point _max) {
        min = _min;
        max = _max;
    };

    Point min;
    Point max;
};

struct Child {
    AABB m_aabb;

    Child(AABB &aabb) {
        m_aabb = aabb;
    }

    Vector getNormal(const Point &A, const Point &B, const Point &C) const {
        Vector v1 = B - A;
        Vector v2 = C - A;
        return v1.cross(v2).normalized();
    }

    Point getPlaneLineIntersection(const Vector &planeNormal, const Point &pointOnPlane, const Point &lineOrigin, const Vector &lineDirection) {
        float denominator = planeNormal.dot(lineDirection);
        if (std::abs(denominator) < 1e-6)
            return lineOrigin;
        float t = planeNormal.dot(pointOnPlane - lineOrigin) / denominator;
        Vector scaled = t * lineDirection;
        return Point(lineOrigin.x + scaled[0], lineOrigin.y + scaled[1], lineOrigin.z + scaled[2]);
    }

    bool isPointInsideAABB(const Point &point) {
        Point min = m_aabb.min;
        Point max = m_aabb.max;
        std::cout << (point.x >= min.x && point.x <= max.x)  << " " << (point.y >= min.y && point.y <= max.y) << " " << (point.z >= min.z && point.z <= max.z) << std::endl;
        return (
            point.x >= min.x && point.x <= max.x &&
            point.y >= min.y && point.y <= max.y &&
            point.z >= min.z && point.z <= max.z)
        ;
    }

    Eigen::VectorXd getRayIntersection(const Point &origin, const Vector &direction) {
        Vector intersect1;
        Vector intersect2;
        bool intersect1Taken = false;
        bool intersect2Taken = false;
        Point min = m_aabb.min;
        Point max = m_aabb.max;

        std::vector<std::tuple<Point, Point, Point>> planes = {
            {min, {max.x, min.y, min.z}, {max.x, max.y, min.z}}, // XY min
            {min, {min.x, min.y, max.z}, {max.x, min.y, max.z}}, // XZ min
            {min, {min.x, max.y, min.z}, {min.x, max.y, max.z}}, // YZ min
            {{min.x, min.y, max.z}, {max.x, min.y, max.z}, max}, // XY max
            {{max.x, min.y, max.z}, {max.x, min.y, min.z}, max}, // YZ max
            {{min.x, max.y, min.z}, {max.x, max.y, min.z}, max}  // XZ max
        };

        auto processIntersection = [&](const Point &A, const Point &B, const Point &C) {
            Point intersect = getPlaneLineIntersection(getNormal(A, B, C), A, origin, direction);
            std::cout << "intesection point: " << intersect.toString() << std::endl;
            if (isPointInsideAABB(intersect)) {
                if (!intersect1Taken) {
                    intersect1 = Vector(intersect.x, intersect.y, intersect.z);
                    intersect1Taken = true;
                } else if (!intersect2Taken) {
                    intersect2 = Vector(intersect.x, intersect.y, intersect.z);
                    intersect2Taken = true;
                }
            }
        };

        for (size_t i = 0; i < planes.size(); ++i) {
            const Point &A = std::get<0>(planes[i]);
            const Point &B = std::get<1>(planes[i]);
            const Point &C = std::get<2>(planes[i]);
            processIntersection(A, B, C);
        }

        if (intersect1Taken && intersect2Taken) {
            return (intersect1 + intersect2) / 2;
        }

        return Eigen::VectorXd();
    }

    Eigen::VectorXd getRayIntersection2(const Point &origin, const Vector &direction) {
        Point min = m_aabb.min;
        Point max = m_aabb.max;

        double tmin = (min.x - origin.x) / direction[0];
        double tmax = (max.x - origin.x) / direction[0];
        if (tmin > tmax) std::swap(tmin, tmax);

        double tymin = (min.y - origin.y) / direction[1];
        double tymax = (max.y - origin.y) / direction[1];
        if (tymin > tymax) std::swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax))
            return Eigen::VectorXd();  // No intersection

        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;

        double tzmin = (min.z - origin.z) / direction[2];
        double tzmax = (max.z - origin.z) / direction[2];
        if (tzmin > tzmax) std::swap(tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax))
            return Eigen::VectorXd();  // No intersection

        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;

        if (tmin < 0 && tmax < 0)
            return Eigen::VectorXd();  // Intersection is behind the ray

        Eigen::Vector3d originVec(origin.x, origin.y, origin.z);  // Convert Point to Vector
        Eigen::Vector3d intersect1 = originVec + tmin * direction;
        Eigen::Vector3d intersect2 = originVec + tmax * direction;

        Eigen::VectorXd result(6);
        result << intersect1.x(), intersect1.y(), intersect1.z(),
                  intersect2.x(), intersect2.y(), intersect2.z();
        return result;
    }
};



int main() {
    // simple case
    // Point startingPoint(0.0, 0.0, 0.0);
    // Vector direction(1.0, 0.0, 0.0);
    // Point minPoint(1, -1, -1);
    // Point maxPoint(3, 1, 1);
    // AABB aabb(minPoint, maxPoint);

    // Point startingPoint(-1.57549, 3.16861, 5.71639);
    // Vector direction(0.399158, -0.721012, 0.566405);
    // Point minPoint(-0.435039, -0.170842, 7.21287);
    // Point maxPoint(0.283361, 0.254199, 8.19165);

    Point startingPoint(0.111875, 0.028752, 3.92247);
    Vector direction(-0.0975472, 0.693474, 0.713847);
    Point minPoint(-0.435039, 2.80445, 6.23409);
    Point maxPoint(0.283361, 3.22949, 7.21287);
    AABB aabb(minPoint, maxPoint);

    Child ch(aabb);

    auto itersection = ch.getRayIntersection2(startingPoint, direction);
    // std::cout << itersection.size();
    std::cout << itersection[0] << ", " << itersection[1] << ", " << itersection[2];

    return 0;
}
