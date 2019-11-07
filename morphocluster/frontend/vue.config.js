module.exports = {
    publicPath: "/frontend",
    devServer: {
        proxy: {
            '/labeling': {
                target: 'http://localhost:5000',
                ws: true,
                changeOrigin: true
            },
            '/static': {
                target: 'http://localhost:5000',
                ws: true,
                changeOrigin: true
            },
            '/api': {
                target: 'http://localhost:5000',
                ws: true,
                changeOrigin: true
            },
            '/get_obj_image': {
                target: 'http://localhost:5000',
                ws: true,
                changeOrigin: true
            }
        }
    }
}
