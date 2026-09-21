#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include <memory>

#include "interval.hpp"
#include "ray.hpp"

class material;

class hit_record{
    public:
        point3 p;
        vec3 normal;
        double t;
        bool front_face;
        std::shared_ptr<material> mat;

        void set_face_normal(const ray &r, const vec3& outward_normal){
            front_face = r.direction().dot(outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

class hittable{
    public:
        virtual ~hittable() = default;
        
        virtual bool hit(const ray& r, const interval &ray_t, hit_record& rec) const = 0;
};


#endif
