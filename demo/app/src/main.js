import Vue from 'vue'
import App from './App.vue'
import axios from 'axios';

//axios.defaults.baseURL = "http://localhost:8081"
axios.defaults.baseURL = "https://autocaption-api.herokuapp.com/"

Vue.config.productionTip = false
Vue.config.devtools = true

new Vue({
  render: h => h(App),
}).$mount('#app')
