<template>
    <div class="container">{{ project }}</div>
</template>

<script>
import { EventBus } from "@/event-bus.js";
import * as api from "@/helpers/api.js";

export default {
    name: "project",
    props: { project_id: Number },
    components: {},
    data() {
        return {
            project: null
        };
    },
    methods: {},
    mounted() {
        // Load node info
        api.getProject(this.project_id)
            .then(data => {
                this.project = data;
                console.log(data);

                EventBus.$emit("set-title", this.project.name);
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style></style>
