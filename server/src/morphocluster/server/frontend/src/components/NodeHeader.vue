<template>
    <div>
        {{JSON.stringify(this.node)}}
    </div>
</template>

<script>
export default {
    name: "node-header",
    props: ["node"],
    mounted () {
        console.log(this.node);
    },
}
</script>
