import Vue from 'vue'
import App from './App.vue'
import axios from 'axios';

axios.defaults.baseURL = "http://localhost:8081"

Vue.config.productionTip = false
Vue.config.devtools = true

new Vue({
  render: h => h(App),
}).$mount('#app')
