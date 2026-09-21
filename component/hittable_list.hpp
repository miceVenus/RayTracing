#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP

#include <memory>
#include <vector>

#include "hittable.hpp"

class hittable_list : public hittable{
    public:
        std::vector<std::shared_ptr<hittable>> objects;

        hittable_list(){

        }

        hittable_list(std::shared_ptr<hittable> object){
            add(object);
        }

        void add(std::shared_ptr<hittable> object){
            objects.emplace_back(object);
        }

        void clear() { objects.clear(); }

        bool hit(const ray& r, const interval &ray_t, hit_record& rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;

            double closest_so_far = ray_t.max;

            for(auto &obj : objects){
                if(obj->hit(r, interval(ray_t.min, closest_so_far), temp_rec)){
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }
};

#endif