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
export default {
  name: "HomeView",
  mounted() {
    this.setBreadcrumbs([{ name: "home" }]);
  },
}
</script>

<style></style>