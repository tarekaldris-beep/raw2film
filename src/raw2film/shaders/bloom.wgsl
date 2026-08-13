struct Kernel {
    weights: array<vec4<f32>>,
};

struct BloomParams {
    threshold: f32,
    softness: f32,
    intensity: f32,
    red: f32,
    green: f32,
    blue: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0)
var src : texture_2d<f32>;

@group(0) @binding(1)
var dst : texture_storage_2d<rgba16float, write>;

@group(0) @binding(2)
var<storage, read> kernel : Kernel;

@group(0) @binding(3)
var<uniform> kernel_size : vec2<u32>;

@group(0) @binding(4)
var<uniform> params : BloomParams;

fn luminance(c : vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn smoothstep01(x : f32) -> f32 {
    let t = clamp(x, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>
) {
    let image_size = vec2<i32>(textureDimensions(src));
    let pixel_coords = vec2<i32>(gid.xy);

    // Early exit if out of bounds
    if (pixel_coords.x >= image_size.x || pixel_coords.y >= image_size.y) {
        return;
    }

    let k_size = vec2<i32>(kernel_size);
    let k_center = k_size / 2;

    // Calculate the top-left starting coordinate in the source image for this pixel
    let start_coord = pixel_coords - k_center;

    // Pre-calculate image limits for clamping
    let max_coord = image_size - vec2<i32>(1);

    let center = textureLoad(src, pixel_coords, 0);

    var glow = vec3<f32>(0.0);
    var weight_idx = 0u;

    for (var ky: i32 = 0; ky < k_size.y; ky++) {
        // Pre-clamp Y coordinate for the entire row iteration
        let src_y = clamp(start_coord.y + ky, 0, max_coord.y);

        for (var kx: i32 = 0; kx < k_size.x; kx++) {
            // Clamp X coordinate
            let src_x = clamp(start_coord.x + kx, 0, max_coord.x);

            let neighbor = textureLoad(src, vec2<i32>(src_x, src_y), 0);
            let weight = kernel.weights[weight_idx];

            // Pixels above the threshold emit; the kernel is an untinted blur.
            let emit_val = smoothstep01(
                (luminance(neighbor.rgb) - params.threshold) / max(params.softness, 1e-6)
            );
            glow += neighbor.rgb * emit_val * weight.rgb;
            weight_idx++;
        }
    }

    // Classic bloom: add the tinted glow back on top of the image.
    let factors = params.intensity * vec3<f32>(params.red, params.green, params.blue);
    let result = center.rgb + glow * factors;

    textureStore(dst, pixel_coords, vec4<f32>(result, center.a));
}
