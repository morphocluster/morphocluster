<template>
    <div class="message-log">
        <p v-for="m in messages" :key="m">{{m}}</p>
    </div>
</template>

<script>
// TODO: Keep overflow div scrolled to bottom
// https://stackoverflow.com/a/21067431/1116842
export default {
    name: "message-log",
    props: ["messages"],
    mounted () {
        //console.log("MessageLog mounted.");
    }
}
</script>

<style>
.message-log {
    height: 2em;
    overflow-y: auto;
}
</style>
