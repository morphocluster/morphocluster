import Vue from 'vue'
import App from './App.vue'
import router from './router'

import "@mdi/font/css/materialdesignicons.css";
import "roboto-fontface/css/roboto/roboto-fontface.css";

// Custom styles
import './assets/styles.css'

import vuetify from './plugins/vuetify';

Vue.config.productionTip = false;
Vue.config.debug = true;

new Vue({
  router,
  vuetify,
  render: h => h(App),
}).$mount('#app')
