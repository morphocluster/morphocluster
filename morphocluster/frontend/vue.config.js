module.exports = {
  "publicPath": "/frontend",
  "devServer": {
    "proxy": "http://localhost:5000"
  },
  "transpileDependencies": [
    "vuetify"
  ]
}