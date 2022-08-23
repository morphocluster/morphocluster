<template>
    <div id="object">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark text-light">
            <router-link class="navbar-brand text-light" to="/"
                >MorphoCluster</router-link
            >
            <ul class="navbar-nav mr-auto">
                <li class="nav-item nav-link text-light">Object</li>
                <li class="nav-item nav-link active text-light">
                    {{ object.object_id }}
                </li>
            </ul>
            <dark-mode-control />
        </nav>
        <div class="scrollable">
            <div class="container">
                <img :src="image_url" :alt="object.object_id" />
                <v-expansion-panels>
                    <v-expansion-panel
                        v-for="n in object.nodes"
                        :key="n.node_id"
                    >
                        <v-expansion-panel-header>
                            Project {{ n.project_name }} ({{ n.project_id }}):
                            {{ n.name }} ({{ n.node_id }})
                        </v-expansion-panel-header>
                        <v-expansion-panel-content>
                            <pre>{{ n }}</pre>
                            <v-btn
                                color="success"
                                class="ma-2 white--text"
                                @click="create_cluster(n)"
                            >
                                <v-icon dark> mdi-plus </v-icon>
                                Create Cluster
                            </v-btn>
                        </v-expansion-panel-content>
                    </v-expansion-panel>

                    <v-expansion-panel>
                        <v-expansion-panel-header>
                            Data
                        </v-expansion-panel-header>
                        <v-expansion-panel-content>
                            <pre>{{ object }}</pre>
                        </v-expansion-panel-content>
                    </v-expansion-panel>
                </v-expansion-panels>
            </div>
        </div>
    </div>
</template>

<script>
import axios from "axios";
import { EventBus } from "@/event-bus.js";

import * as api from "@/helpers/api.js";

export default {
    name: "object",
    props: ["object_id"],
    components: {},
    data() {
        return {
            object: null,
        };
    },
    methods: {
        create_cluster(node) {
            console.log(
                `Creating node for object ${this.object.object_id} in project ${node.project_id}...`
            );

            api.createNode({
                project_id: node.project_id,
                parent_id: node.node_id,
                members: [{ object_id: this.object_id }],
            })
                .then((node) => {
                    console.log("Created node", node);
                })
                .catch((e) => {
                    this.axiosErrorHandler(e);
                });
        },
    },
    mounted() {
        // Load node info
        axios
            .get(`/api/objects/${this.object_id}`)
            .then((response) => {
                this.object = response.data;
                console.log(response.data);

                EventBus.$emit("set-title", this.object.object_id);
            })
            .catch((e) => {
                console.log(e);
            });
    },
    computed: {
        image_url: function () {
            return `/get_obj_image/${this.object.object_id}`;
        },
    },
};
</script>

<style>
#object {
    height: 100%;
}
.scrollable {
    height: 100%;
    overflow: auto;
}
#action {
    margin: 0 auto;
}
</style>
