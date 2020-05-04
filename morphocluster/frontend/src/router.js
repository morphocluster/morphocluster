import Vue from 'vue'
import Router from 'vue-router'

import NotFound from './views/NotFound.vue'
import * as api from "@/helpers/api.js";

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
      name: 'grow',
      path: '/projects/:project_id/grow/:node_id?',
      component: () => import(/* webpackChunkName: "grow" */ './views/Grow.vue'),
    },
    {
      name: 'labeling2',
      path: '/labeling2/:node_id',
      component: () => import(/* webpackChunkName: "labeling2" */ './views/Labeling2.vue'),
      props: (route) => ({ node_id: parseInt(route.params.node_id) }),
    },
    {
      name: 'validate',
      path: '/projects/:project_id/validate/:node_id?',
      component: () => import(/* webpackChunkName: "validate" */ './views/Validate.vue'),
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
      name: 'datasets-create',
      path: '/datasets/create',
      beforeEnter: (to, from, next) => {
        api.createDataset().then((dataset) => {
          next({
            name: "dataset-edit",
            params: { dataset_id: dataset.dataset_id }
          })
        }).catch(e => {
          console.log(e);
        });
      }
    },
    {
      name: 'dataset',
      path: '/datasets/:dataset_id',
      component: () => import(/* webpackChunkName: "datasets" */ './views/Dataset.vue'),
      props: (route) => ({ dataset_id: parseInt(route.params.dataset_id) }),
    },
    {
      name: 'dataset-edit',
      path: '/datasets/:dataset_id/edit',
      component: () => import(/* webpackChunkName: "datasets" */ './views/DatasetEdit.vue'),
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
