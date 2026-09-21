#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <vector>
#include <future>
#include <cstdint>
#include <utility>
#include "ray.hpp"
#include "color.hpp"
#include "interval.hpp"
#include "material.hpp"
#include "hittable.hpp"
#include "thread_pool.hpp"
#include "../render/image.hpp"

#include "raytracing.hpp"

class camera{
    public:

        double aspect_ratio = 1.0;
        int image_width = 100;
        int samples_per_pixel = 10;
        int max_depth = 10;

        double vfov = 90.0;
        point3 lookfrom = point3(0, 0, 0);
        point3 lookat   = point3(0, 0, -1);
        vec3   vup      = vec3(0, 1, 0);

        double defocus_angle    = 0;
        double focus_dist       = 10;

        explicit camera(int thread_count = 16, std::uint64_t seed = 1)
            : pool(thread_count > 0 ? static_cast<std::size_t>(thread_count) : 1), seed(seed) {}

        RenderImage render_to_image(const hittable &world){
            initialize();

            RenderImage image;
            image.width = image_width;
            image.height = image_height;
            image.rgba.resize(static_cast<size_t>(image_width) * image_height * 4);

            std::vector<std::future<void>> results;
            results.reserve(static_cast<size_t>(image_height));

            for (int j = 0; j < image_height; j++) {
                results.emplace_back(pool.enqueue([this, j, &world, &image] {
                    for (int i = 0; i < image_width; i++) {
                        const std::size_t pixel = static_cast<std::size_t>(j) * image_width + i;
                        pcg rng(pixel_seed(seed ^ static_cast<std::uint64_t>(pixel)));
                        pcg_scope rng_scope(rng);
                        color pixel_color(0,0,0);
                        for (int sample = 0; sample < samples_per_pixel; sample++) {
                            ray r = this->get_ray(i, j);
                            pixel_color += ray_color(r, world, max_depth);
                        }
                        write_rgba(image.rgba, pixel, pixel_color * pixel_samples_scale);
                    }
                }));
            }

            for (auto &result : results) {
                result.get();
            }
            return image;
        }

    private:

        int image_height;
        double pixel_samples_scale;
        point3 pixel00_loc;
        point3 camera_center;
        vec3 pixel_delta_u;
        vec3 pixel_delta_v;
        vec3 u, v, w;   // x, y ,z
        vec3 defocus_disk_u;
        vec3 defocus_disk_v;
        ThreadPool pool;
        std::uint64_t seed;

        static std::uint64_t pixel_seed(std::uint64_t value) {
            value += 0x9e3779b97f4a7c15ULL;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
            return value ^ (value >> 31);
        }

        void initialize(){

            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            auto theta  = deg2rad(vfov);
            auto h      = std::tan(theta/2);
            auto viewport_height = 2 * h * focus_dist;

            auto viewport_width = viewport_height * (double(image_width)/image_height);
            camera_center = lookfrom;

            w = unit(lookfrom - lookat);
            u = unit(cross(vup, w));
            v = cross(w, u);

            auto viewport_u = u * viewport_width;
            auto viewport_v = v * (-viewport_height);



            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            auto viewport_left_up = camera_center - (w * focus_dist) - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_left_up + (pixel_delta_u + pixel_delta_v) * 0.5;

            auto defocus_radius = focus_dist * std::tan(deg2rad(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }

        color ray_color(const ray &r, const hittable &world, int depth){
            if(depth <= 0) return color(0, 0, 0);

            hit_record rec;
            if(world.hit(r, interval(0.001, infinity), rec)){
                ray scattered;
                color attenuation;
                if(rec.mat->scatter(r, rec, attenuation, scattered)){
                    return attenuation * ray_color(scattered, world, depth-1);
                }else
                    return color(0, 0, 0);
            }

            vec3 u_d = unit(r.direction());
            auto a = 0.5 * (u_d.y() + 1.0);
            return (color(1.0, 1.0, 1.0) * (1.0-a)) + (color(0.5, 0.7, 1.0) * a);
        }

        ray get_ray(int i, int j) const{
            auto offset = sample_square();
            auto pixel_sample = pixel00_loc + 
                                (pixel_delta_u * (i + offset.x())) + 
                                (pixel_delta_v * (j + offset.y()));

            auto ray_origin = defocus_angle == 0 ? camera_center : defocus_disk_sample();
            auto ray_direction = pixel_sample - ray_origin;

            return ray(ray_origin, ray_direction);
        }

        vec3 sample_square() const{
            return vec3(random_double() - 0.5, random_double() - 0.5, 0);
        }

        point3 defocus_disk_sample() const{
            auto p = random_in_unit_disk();

            return camera_center + (defocus_disk_u * p[0]) + (defocus_disk_v * p[1]);
        }

        static void write_rgba(std::vector<std::uint8_t> &rgba, size_t pixel, const color& pixel_color){
            auto r = linear_to_gamma(pixel_color.x());
            auto g = linear_to_gamma(pixel_color.y());
            auto b = linear_to_gamma(pixel_color.z());

            static const interval intensity = interval(0.0, 0.999);
            auto offset = pixel * 4;

            rgba[offset] = static_cast<std::uint8_t>(256 * intensity.clamp(r));
            rgba[offset + 1] = static_cast<std::uint8_t>(256 * intensity.clamp(g));
            rgba[offset + 2] = static_cast<std::uint8_t>(256 * intensity.clamp(b));
            rgba[offset + 3] = 255;
        }

};

#endif
