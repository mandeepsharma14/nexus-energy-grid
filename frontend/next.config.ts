/** @type {import('next').NextConfig} */
// NexusGrid Frontend — © 2026 Mandeep Sharma. All rights reserved.
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
};
export default nextConfig;
