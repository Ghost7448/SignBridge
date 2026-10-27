import { defineConfig } from "vite";
export default defineConfig({base:"./",publicDir:"static",build:{outDir:"www",emptyOutDir:true,target:"es2020"}});
