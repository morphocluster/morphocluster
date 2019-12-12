<template>
    <div id="datasets">
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
                <h1>Datasets</h1>
                <b-table
                    id="datasets_table"
                    striped
                    sort-by="name"
                    :items="datasets"
                    :fields="fields"
                    showEmpty
                >
                    <template v-slot:cell(name)="data">
                        <router-link
                            :to="{
                                name: 'dataset',
                                params: { dataset_id: data.item.dataset_id }
                            }"
                            >{{ data.item.name }}</router-link
                        >
                    </template>
                    <template slot="empty">
                        <div class="text-center">No datasets available.</div>
                    </template>
                </b-table>
                <!--<b-button
                    size="sm"
                    variant="success"
                    class="mr-2 float-right"
                    :to="{name: 'datasets-add'}"
                >
                    <i class="mdi mdi-plus" /> Add dataset
                </b-button>-->
            </div>
        </div>
    </div>
</template>

<script>
import "@mdi/font/css/materialdesignicons.css";
import * as api from "@/helpers/api.js";

export default {
    name: "datasets",
    props: {},
    components: {},
    data() {
        return {
            fields: [
                { key: "name", sortable: true }
                //"action"
            ],
            datasets: [],
            alerts: [],
            breadcrumb: [
                {
                    text: "Datasets"
                }
            ]
        };
    },
    methods: {},
    mounted() {
        api.datasetsGetAll()
            .then(datasets => {
                this.datasets = datasets;
            })
            .catch(e => {
                console.log(e);
                this.alerts.unshift({
                    message: e.message,
                    variant: "danger"
                });
            });
    }
};
</script>

<style>
#datasets {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}

#datasets_table tr td:nth-child(1) {
    width: 100%;
}

#datasets_table tr td:not(:nth-child(1)) {
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
