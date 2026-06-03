/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  eslint: {
    ignoreDuringBuilds: false,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  experimental: {
    optimizePackageImports: ["@/components", "@/lib", "@/utils"],
  },
  compress: true,
  images: {
    formats: ["image/webp", "image/avif"],
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Content-Security-Policy", value: "frame-ancestors 'none'" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
  async rewrites() {
    const backend = process.env.API_INTERNAL_URL || "http://localhost:8000";
    return [
      { source: "/auth/:path*", destination: `${backend}/auth/:path*` },
      { source: "/memories/:path*", destination: `${backend}/memories/:path*` },
      { source: "/configure/:path*", destination: `${backend}/configure/:path*` },
      { source: "/configure", destination: `${backend}/configure` },
      { source: "/reset", destination: `${backend}/reset` },
      { source: "/generate-instructions", destination: `${backend}/generate-instructions` },
      { source: "/api-keys/:path*", destination: `${backend}/api-keys/:path*` },
      { source: "/api-keys", destination: `${backend}/api-keys` },
      { source: "/requests", destination: `${backend}/requests` },
      { source: "/entities/:path*", destination: `${backend}/entities/:path*` },
    ];
  },
  redirects: async () => {
    return [
      {
        source: "/settings",
        destination: "/dashboard/settings",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
