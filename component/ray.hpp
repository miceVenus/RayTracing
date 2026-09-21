#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"


class ray{


    public:

        ray(){

        }
        
        ray(const point3& ori, const vec3 &dir) : ori(ori), dir(dir){
        }

        const vec3 &direction() const{
            return dir;
        }

        const point3 &origin() const{
            return ori;
        }

        point3 at(double t) const{
            return ori + dir * t;
        }




    private:

        point3 ori;
        vec3 dir;
};


#endif