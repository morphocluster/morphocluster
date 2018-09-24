<template>
  <div id="app">
    <router-view />
  </div>
</template>

<script>
import { EventBus } from '@/event-bus.js';

export default {
  data() {
    return {
      title: undefined,
    }
  },
  mounted () {
    EventBus.$on('set-title', title => {
      this.title = title;
    });
  }
}
</script>


<style>
#app{
  height: 100%;

  display: flex;
  flex-direction: column;
  /*align-items: stretch;*/

  overflow: hidden;
}
</style>
