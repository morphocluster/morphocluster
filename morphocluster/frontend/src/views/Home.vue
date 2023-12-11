<template>
  <div id="home">
    <div class="container">
      <h1>MorphoCluster</h1>
      <b-button size="m" variant="primary" class="mr-2" :to="{
        name: 'projects'
      }">
        Projects
      </b-button>
      <b-button size="l" variant="primary" class="mr-2" :to="{
        name: 'files'
      }">
        Files
      </b-button>
    </div>
  </div>
</template>

<script>
export default {
  name: "ProjectsView",
}
</script>

<style></style>