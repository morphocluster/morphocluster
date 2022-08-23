import Vue from 'vue'
import App from './App.vue'
import router from './router'

// Bootstrap
import BootstrapVue from 'bootstrap-vue'
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'
Vue.use(BootstrapVue);

// Custom styles
import './assets/styles.css'

import vuetify from './plugins/vuetify'

Vue.config.productionTip = false

new Vue({
  router,
  render: h => h(App),
  vuetify,

  data() {
    return {
      config: window.config,
    };
  }
}).$mount('#app')
