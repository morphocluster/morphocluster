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
      path: '/projects/:project_id/bisect/:node_id?',
      component: () => import(/* webpackChunkName: "bisect" */ './views/Bisect.vue'),
    },
    {
      name: 'labeling2',
      path: '/labeling2/:node_id',
      component: () => import(/* webpackChunkName: "labeling2" */ './views/Labeling2.vue'),
      props: (route) => ({ node_id: parseInt(route.params.node_id) }),
    },
    {
      name: 'approve',
      path: '/projects/:project_id/approve/:node_id?',
      component: () => import(/* webpackChunkName: "approve" */ './views/Approve.vue'),
    },
    {
      name: 'project',
      path: '/projects/:project_id',
      component: () => import(/* webpackChunkName: "project" */ './views/Project.vue'),
      props: (route) => ({ project_id: parseInt(route.params.project_id), dataset_id: parseInt(route.params.dataset_id) }),
    },
    {
      name: 'datasets',
      path: '/datasets',
      component: () => import(/* webpackChunkName: "datasets" */ './views/Datasets.vue'),
    },
    {
      name: 'datasets-add',
      path: '/datasets/add',
      component: () => import(/* webpackChunkName: "datasets-add" */ './views/DatasetsAdd.vue'),
    },
    {
      name: 'dataset',
      path: '/datasets/:dataset_id',
      component: () => import(/* webpackChunkName: "dataset" */ './views/Dataset.vue'),
      props: (route) => ({ dataset_id: parseInt(route.params.dataset_id) }),
    },
    {
      path: '/',
      redirect: '/datasets'
    },
    { path: '*', component: NotFound }
  ]
});

export default router;
