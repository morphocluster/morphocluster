<template>
    <v-container id="project">
        <template v-if="project">
            <h2>{{project.name}}</h2>

            <v-row class="my-2">
                <v-btn
                    small
                    class="mr-2"
                    :to="{
                        name: 'validate',
                        params: { project_id: project.project_id }
                    }"
                >Validate</v-btn>
                <v-btn
                    small
                    class="mr-2"
                    :to="{
                        name: 'grow',
                        params: { project_id: project.project_id }
                    }"
                >Grow</v-btn>
            </v-row>

            <v-row class="my-2">
                <v-expansion-panels>
                    <v-expansion-panel>
                        <v-expansion-panel-header>Raw</v-expansion-panel-header>
                        <v-expansion-panel-content>
                            <code class="d-block mx-2">{{ project }}</code>
                        </v-expansion-panel-content>
                    </v-expansion-panel>
                </v-expansion-panels>
            </v-row>
        </template>
    </v-container>
</template>


<script>
import * as api from "@/helpers/api.js";
import mixins from "@/mixins.js";

export default {
    name: "project",
    props: { project_id: Number },
    components: {},
    mixins: [mixins],
    data() {
        return {
            project: null
        };
    },
    methods: {},
    created() {
        // Load node info
        this.setLoading("project");
        api.getProject(this.project_id, true, true)
            .then(data => {
                this.project = data;
                console.log(data);

                // Update breadcrumb
                this.setBreadcrumbs([
                    {
                        text: this.project.dataset.name,
                        to: {
                            name: "dataset",
                            params: {
                                dataset_id: this.project.dataset_id
                            }
                        },
                        exact: true
                    },
                    {
                        text: this.project.name,
                        to: {
                            name: "project",
                            params: { project_id: this.project.project_id }
                        }
                        //exact: true
                    }
                ]);
            })
            .catch(e => {
                console.log(e);
            })
            .finally(() => {
                this.unsetLoading("project");
            });
    }
};
</script>

<style></style>
