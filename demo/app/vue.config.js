module.exports = {
  //Remove publicPath if deploying locally
  publicPath: '/autocaption/',
  configureWebpack: {
    devServer: {
      watchOptions: {
        poll: true
      }
    }
  }
}