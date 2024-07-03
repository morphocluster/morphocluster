module.exports = {
  "publicPath": "/frontend",
  "devServer": {
    publicPath: "/frontend",
    proxy: {
      '/labeling': {
        target: 'http://localhost:5000',
        ws: true,
        changeOrigin: true
      },
      '/config.js': {
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
  },
  "transpileDependencies": [
    "vuetify"
  ],
  css: {
    loaderOptions: {
      sass: {
        sassOptions: {
          quietDeps: true
        }
      }
    }
  }
}