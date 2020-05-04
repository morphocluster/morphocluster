<template>
    <div id="dataset" class="view">
        <div class="container">
            <div class="alerts" v-if="alerts.length">
                <b-alert
                    :key="a"
                    v-for="a of alerts"
                    dismissible
                    show
                    :variant="a.variant"
                >{{ a.message }}</b-alert>
            </div>
            <h2>Properties</h2>
            <v-row class="my-2">
                <v-expansion-panels>
                    <v-expansion-panel>
                        <v-expansion-panel-header>Raw</v-expansion-panel-header>
                        <v-expansion-panel-content>
                            <code class="d-block mx-2">{{ dataset }}</code>
                        </v-expansion-panel-content>
                    </v-expansion-panel>
                </v-expansion-panels>
            </v-row>
            <h2>Import objects</h2>
            <h2>Import features</h2>
        </div>
    </div>
</template>

<script>
import * as api from "@/helpers/api.js";
import mixins from "@/mixins.js";

export default {
    name: "dataset-edit",
    props: { dataset_id: Number },
    mixins: [mixins],
    data() {
        return {
            dataset: null,
            alerts: []
        };
    },
    methods: {},
    created() {
        // Load node info
        api.getDataset(this.dataset_id)
            .then(data => {
                this.dataset = data;

                // Update breadcrumb
                this.setBreadcrumbs([
                    {
                        text: this.dataset.name,
                        to: {
                            name: "dataset",
                            props: { dataset_id: this.dataset.dataset_id }
                        },
                        exact: true
                    },
                    { text: "Edit" }
                ]);
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style>
</style>
