<template>
  <div id="home">
    <div class="container">
      <v-btn large color="primary" class="mr-2" :to="{ name: 'projects' }">
        Projects
      </v-btn>
      <v-btn large color="primary" class="mr-2" :to="{ name: 'files' }">
        Files
      </v-btn>
    </div>
  </div>
</template>


<script>

import state from "../globalState.js";

export default {
  name: "ProjectsView",
  methods: {
    async initialize() {
      state.setBreadcrumbs([], "home");
    }
  },
  mounted() {
    this.initialize();
  },
}
</script>

<style></style>