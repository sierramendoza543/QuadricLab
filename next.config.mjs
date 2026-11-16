/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["three"],
  // lets a second build or preview run without touching a running dev server's .next folder
  distDir: process.env.NEXT_DIST_DIR || ".next",
};

export default nextConfig;
