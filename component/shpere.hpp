#ifndef SHPERE_HPP
#define SHPERE_HPP

#include "hittable.hpp"
#include "vec3.hpp"

class sphere : public hittable{
    public:
        sphere(const point3& center, const double radius, const std::shared_ptr<material> mat) : 
        center(center), radius(std::fmax(0, radius)), mat(mat){

        }

        bool hit(const ray& r, const interval &ray_t, hit_record& rec) const override{
            
            vec3 c2c = center - r.origin();

            auto a = r.direction().squard_sum();
            auto h = r.direction().dot(c2c);
            auto c = c2c.squard_sum() - (radius * radius);

            auto discriminant = h*h - a*c;

            if(discriminant < 0) return false;

            auto sqrtd = std::sqrt(discriminant);
            auto root = (h - sqrtd) / a;

            if(!ray_t.surrounds(root)){
                root = (h + sqrtd) / a;
                if(!ray_t.surrounds(root)) return false;
            }

            rec.p = r.at(root);
            rec.normal = (rec.p - center) / radius;
            rec.set_face_normal(r, rec.normal);
            rec.t = root;
            rec.mat = mat;
            return true;
        }

    private:
        point3 center;
        double radius;
        std::shared_ptr<material> mat;
};

#endif
