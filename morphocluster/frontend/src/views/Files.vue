<template>
  <div id="files">
    <nav class="navbar navbar-expand-lg navbar-light bg-dark text-light">
          <ul class="navbar-nav mr-5">
                <li class="nav-item active text-light">MorphoCluster</li>
            </ul>
            <router-link class="navbar-brand text-light mr-5" to="/"
                >Projects</router-link
            >
            <router-link class="navbar-brand text-light mr-auto mr-5" to="Files"
                >Files</router-link
            >
    </nav>
  </div>
</template>

<script>

export default {
    name: "FilesView",
    props: {},
    components: {}
}




</script>

<style>
#files {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}

#projects_table tr td:nth-child(1) {
    width: 100%;
}

#projects_table tr td:not(:nth-child(1)) {
    width: auto;
    text-align: right;
    white-space: nowrap;
}

.scrollable {
    overflow-y: auto;
}

.alerts {
    padding-top: 1em;
}
</style>