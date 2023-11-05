
<template>
  <div id="file">
    <nav class="navbar navbar-expand-lg navbar navbar-dark bg-dark">
      <a class="navbar-brand" href="/p">MorphoCluster</a>
      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav"
        aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav">
          <li class="nav-item">
            <a class="nav-link" href="/p">Projects</a>
          </li>
          <li class="nav-item">
            <router-link class="nav-link" :to="{ name: 'files', params: { file_path: '' }, }">Files</router-link>
          </li>
        </ul>
      </div>
    </nav>
    <div class="container">
      <b-table id="files_table" striped sort-by="name" :items="file" :fields="fields" showEmpty>
      </b-table>
    </div>
    <div class="d-flex justify-content-center">
      <b-button size="sm" variant="primary" class="mx-2">
        Download
      </b-button>
      <b-button size="sm" variant="primary" class="mx-2">
        Save as project
      </b-button>
    </div>
  </div>
</template>



<script>

import axios from "axios";
import "bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";

export default {
  name: "FileView",
  props: { file_name: String, file_path: String },
  components: {},
  data() {
    return {
      fields: [
        { key: "name", sortable: true },
        "last_modified",
        "path",
        "type",
      ],
      file: [],
      alerts: [],
    };
  },
  mounted() {

  },
  created() {
    this.initialize();
  },
  watch: {
    $route: "initialize",
  },
  methods: {
    async initialize() {
      try {
        const response = await axios.get(`/api/file/${this.file_path}`);
        this.file = response.data;
      } catch (error) {
        console.error(error);
        this.alerts.unshift({
          message: error.message,
          variant: "danger",
        });
      }
    },
  },

};
</script>

<style></style>
