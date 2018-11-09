import Vue from 'vue'
import Router from 'vue-router'

import NotFound from './views/NotFound.vue'

Vue.use(Router)

var router = new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      name: 'labeling',
      path: '/labeling',
      redirect: function () { window.location.reload(); }
    },
    {
      name: 'bisect',
      path: '/p/:project_id/bisect/:node_id?',
      component: () => import(/* webpackChunkName: "bisect" */ './views/Bisect.vue'),
    },
    {
      name: 'labeling2',
      path: '/labeling2/:node_id',
      component: () => import(/* webpackChunkName: "labeling2" */ './views/Labeling2.vue'),
      props: (route) => ({ node_id: parseInt(route.params.node_id) }),
    },
    {
      name: 'projects',
      path: '/p',
      component: () => import(/* webpackChunkName: "projects" */ './views/Projects.vue'),
    },
    {
      name: 'approve',
      path: '/p/:project_id/approve/:node_id?',
      component: () => import(/* webpackChunkName: "approve" */ './views/Approve.vue'),
    },
    {
      name: 'project',
      path: '/p/:project_id',
      component: () => import(/* webpackChunkName: "project" */ './views/Project.vue'),
      props: (route) => ({ project_id: parseInt(route.params.project_id) }),
    },
    {
      path: '/',
      redirect: '/p'
    },
    { path: '*', component: NotFound }
  ]
});

export default router;
