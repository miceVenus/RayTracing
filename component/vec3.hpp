#ifndef VEC3_HPP
#define VEC3_HPP

#include "raytracing.hpp"


class vec3{

    public:
        double e[3];

        vec3() : e{0, 0, 0}{}
        vec3(double x, double y, double z) : e{x, y ,z}{};

        double x() const {return e[0]; }
        double y() const {return e[1]; }
        double z() const {return e[2]; }

        vec3 operator-() const{
            return vec3(-e[0], - e[1], -e[2]);
        }

        vec3 operator-(const vec3 &v) const{
            return vec3(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]);
        }

        double operator[](int i) const{
            return e[i];
        }

        double &operator[](int i) {
            return e[i];
        }

        vec3 &operator+=(const vec3 &v){
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];

            return *this;
        }

        vec3 operator+(const vec3 &v) const{
            return vec3(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]);
        }

        vec3 &operator*=(double t){
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;

            return *this;
        }

        vec3 operator*(const vec3 &v) const{
            return vec3(e[0] * v.e[0], e[1] * v.e[1], e[2] * v.e[2]);
        }

        vec3 operator*(double t) const{
            return vec3(e[0] * t, e[1] * t, e[2] * t);
        }

        vec3 &operator/=(double t){
            e[0] /= t;
            e[1] /= t;
            e[2] /= t;

            return *this;
        }

        vec3 operator/(double t) const{
            return vec3(e[0] / t, e[1] / t, e[2] / t);
        }

        double length() const{
            return std::sqrt(squard_sum());
        }

        double squard_sum() const{
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }

        double dot(const vec3 &v) const{
            return e[0] * v.e[0] + e[1] * v.e[1] + e[2] * v.e[2];
        }

        static vec3 random() {
            return vec3(random_double(), random_double(), random_double());
        }

        static vec3 random(double min, double max){
            return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
        }

        std::ostream& operator<<(std::ostream& out){
            return out << e[0] << ' ' << e[1] << ' ' << e[2];
        }
};

using point3 = vec3;

// inline vec3 operator/(const vec3 &v, const double t){
//     return v / t;
// }


inline vec3 reflect(const vec3 &v, const vec3 &n){
    return v - n * (v.dot(n) * 2);
}

inline vec3 refract(const vec3 &uv, const vec3 &n, double etai_over_etao){
    auto cos_theta = std::fmin((-uv).dot(n), 1.0);
    vec3 r_out_perp = (uv + n * cos_theta) * etai_over_etao;
    vec3 r_out_norm = n * (-std::sqrt(std::fabs(1.0 - r_out_perp.squard_sum())));
    return r_out_perp + r_out_norm;
}

inline vec3 unit(const vec3 &v){
    return v / v.length();
}

inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3::random(-1,1);
        auto lensq = p.squard_sum();
        if (lensq > 1e-160 && lensq <= 1)
            return p / sqrt(lensq);
    }
}

inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_double(-1,1), random_double(-1,1), 0);
        if (p.squard_sum() < 1)
            return p;
    }
}

inline vec3 random_on_hemishpere(const vec3 &normal){
    vec3 unit_vec3 = random_unit_vector();

    if(unit_vec3.dot(normal) >= 0){
        return unit_vec3;
    }else{
        return -unit_vec3;
    }
}


inline vec3 cross(const vec3 &u, const vec3 &v){
    return vec3(u[1] * v[2] - u[2] * v[1], 
                u[2] * v[0] - u[0] * v[2], 
                u[0] * v[1] - u[1] * v[0]);
}

inline bool near_zero(vec3 &v){
    auto s = 1e-8;
    if((std::fabs(v[0]) < s) && (std::fabs(v[1]) < s) && (std::fabs(v[2]) < s))
        return true;
    
    return false;
}

#endif

