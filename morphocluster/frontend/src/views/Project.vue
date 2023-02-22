<template>
    <div class="container">
         {{project}}
    </div>
</template>

<script>
import axios from "axios";
import { EventBus } from "@/event-bus.js";

export default {
    name: "ProjectView",
    props: {"project_id": Number},
    components: {},
    data() {
        return {
            project: null,
        };
    },
    methods: {

    },
    mounted() {
        // Load node info
        axios
            .get(`/api/projects/${this.project_id}`)
            .then(response => {
                this.project = response.data;
                console.log(response.data);

                EventBus.$emit("set-title", this.project.name);
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style>

</style>
