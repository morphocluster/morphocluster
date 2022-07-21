import Vue from 'vue'
import Router from 'vue-router'

import NotFound from './views/NotFound.vue'

Vue.use(Router)

var router = new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      name: 'datasets',
      path: '/datasets',
      component: () => import(/* webpackChunkName: "projects" */ './views/Datasets.vue'),
    },
    {
      name: 'labeling',
      path: '/labeling',
      redirect: function () { window.location.reload(); }
    },
    {
      name: 'bisect',
      path: '/datasets/:dataset_id/projects/:project_id/bisect/:node_id?',
      component: () => import(/* webpackChunkName: "bisect" */ './views/Bisect.vue'),
      props: (route) => ({ dataset_id: parseInt(route.params.dataset_id) }),
    },
    {
      name: 'labeling2',
      path: '/labeling2/:node_id',
      component: () => import(/* webpackChunkName: "labeling2" */ './views/Labeling2.vue'),
      props: (route) => ({ node_id: parseInt(route.params.node_id) }),
    },

    {
      name: 'projects',
      path: '/datasets/:dataset_id/projects',
      component: () => import(/* webpackChunkName: "projects" */ './views/Projects.vue'),
      props: (route) => ({ dataset_id: parseInt(route.params.dataset_id) }),
    },
    {
      name: 'approve',
      path: '/datasets/:dataset_id/projects/:project_id/approve/:node_id?',
      component: () => import(/* webpackChunkName: "approve" */ './views/Approve.vue'),
      props: (route) => ({ dataset_id: parseInt(route.params.dataset_id) }),
    },
    {
      name: 'project',
      path: '/datasets/:dataset_id/projects/:project_id',
      component: () => import(/* webpackChunkName: "project" */ './views/Project.vue'),
      props: (route) => ({ project_id: parseInt(route.params.project_id) }),
    },
    {
      path: '/',
      redirect: '/datasets'
    },
    { path: '*', component: NotFound }
  ]
});

export default router;
