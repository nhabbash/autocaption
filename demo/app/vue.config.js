module.exports = {
  publicPath: '/autocaption/',
  configureWebpack: {
    devServer: {
      watchOptions: {
        poll: true
      }
    }
  }
}