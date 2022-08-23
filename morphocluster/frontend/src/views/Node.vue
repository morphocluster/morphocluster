<template>
    <div class="container">
         {{node}}
    </div>
</template>

<script>
import axios from "axios";
import { EventBus } from "@/event-bus.js";

export default {
    name: "node",
    props: {"node_id": Number},
    components: {},
    data() {
        return {
            node: null,
        };
    },
    methods: {

    },
    mounted() {
        // Load node info
        axios
            .get(`/api/nodes/${this.node_id}`)
            .then(response => {
                this.node = response.data;
                console.log(response.data);

                EventBus.$emit("set-title", this.node.name);
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style>

</style>
