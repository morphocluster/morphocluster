<template>
    <div id="dataset">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark">
            <router-link class="navbar-brand text-light" to="/"
                >MorphoCluster</router-link
            >
            <b-breadcrumb :items="breadcrumb"></b-breadcrumb>
        </nav>

        <div class="scrollable">
            <div class="container">
                <div class="alerts" v-if="alerts.length">
                    <b-alert
                        :key="a"
                        v-for="a of alerts"
                        dismissible
                        show
                        :variant="a.variant"
                        >{{ a.message }}</b-alert
                    >
                </div>
                <h1 v-if="dataset">{{ dataset.name }}</h1>
                <!--{{dataset}}-->
                <h2>Projects</h2>
                <projects-table :dataset_id="dataset_id" />
                <!--
                <h2>Expert Mode</h2>
                <div style="margin: 0 auto; width: 0;">
                    <b-button size="sm" variant="danger" class="mr-2" href="/labeling">Expert mode</b-button>
                </div>
                -->
            </div>
        </div>
    </div>
</template>

<script>
import * as api from "@/helpers/api.js";
import { EventBus } from "@/event-bus.js";

import ProjectsTable from "@/components/ProjectsTable.vue";

export default {
    name: "dataset",
    props: { dataset_id: Number },
    components: { ProjectsTable },
    data() {
        return {
            dataset: null,
            alerts: [],
            breadcrumb: [
                {
                    text: "Datasets",
                    to: "/datasets"
                }
            ]
        };
    },
    methods: {
        updateBreadcrumb() {
            this.breadcrumb = [
                {
                    text: "Datasets",
                    to: "/datasets"
                },
                {
                    text: this.dataset.name
                }
            ];
        }
    },
    created() {
        // Load node info
        api.getDataset(this.dataset_id)
            .then(data => {
                this.dataset = data;

                this.updateBreadcrumb();

                EventBus.$emit("set-title", this.dataset.name);
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style>
#dataset {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}
</style>
