#ifndef MATERIAL_HPP
#define MATERIAL_HPP


#include "hittable.hpp"
#include "color.hpp"
#include "ray.hpp"

class material{

    public:
        virtual ~material() = default;
        virtual bool scatter(const ray&, const hit_record&, color&, ray&) const {return false;}
};

class lambertian : public material{
    public:
        lambertian(const color &albedo) : albedo(albedo){
        }

        bool scatter(const ray&, const hit_record& rec, color& attenuation, ray& scattered) const override {
            auto scattered_direction = rec.normal + random_unit_vector();
            if(near_zero(scattered_direction)){
                scattered_direction = rec.normal;
            }
            attenuation = albedo;
            scattered = ray(rec.p, scattered_direction);

            return true;
        }

    private:
        color albedo;
};


class metal : public material{
    public:
        metal(const color &albedo, const double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1){

        }

        bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
        const override{
            auto scattered_direction = reflect(r_in.direction(), rec.normal);
            scattered_direction = unit(scattered_direction) + (random_unit_vector() * fuzz);
            attenuation = albedo;
            scattered = ray(rec.p, scattered_direction);

            return (scattered_direction.dot(rec.normal) > 0);
        }

    private:
        color albedo;
        double fuzz;
};

class dielectric : public material {
    public:
        dielectric(const double eta) 
        :eta(eta){

        }

        bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override{
            attenuation = color(1.0, 1.0, 1.0);
            double etai_over_etao = rec.front_face ? 1.0 / eta : eta;

            auto unit_direction = unit(r_in.direction());
            double cos_theta = std::fmin(-unit_direction.dot(rec.normal), 1.0);
            double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

            vec3 scattered_direction;
            bool cannot_refract = (etai_over_etao * sin_theta) > 1.0;
            if (cannot_refract || reflectance(cos_theta, etai_over_etao) > random_double()){
                scattered_direction = reflect(r_in.direction(), rec.normal);
            }else{
                scattered_direction = refract(unit_direction, rec.normal, etai_over_etao);
            }

            scattered = ray(rec.p, scattered_direction);
            return true;
        }

    private:
        double eta;

        static double reflectance(double cosine, double etai_over_etao){
            auto r0 = (1 - etai_over_etao) / (1 + etai_over_etao);
            r0 = r0*r0;
            return r0 + (1-r0)*std::pow((1 - cosine),5);
        }
};

#endif
