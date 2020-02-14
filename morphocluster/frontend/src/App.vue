<template>
    <v-app>
        <v-app-bar app color="grey darken-3" dark>
            <div class="d-flex align-center">
                <router-link to="/">
                    <v-img
                        alt="MorphoCluster Logo"
                        class="shrink mr-2"
                        contain
                        src="@/assets/morphocluster.png"
                        transition="scale-transition"
                        width="40"
                    />
                </router-link>
            </div>

            <v-toolbar-title>
                <router-link to="/">MorphoCluster</router-link>
            </v-toolbar-title>

            <v-breadcrumbs :items="globalState.breadcrumbs" large></v-breadcrumbs>
        </v-app-bar>
        <v-content>
            <v-tooltip top>
                <template v-slot:activator="{ on }">
                    <v-progress-linear
                        v-on="on"
                        :active="globalState.loading.length"
                        absolute
                        indeterminate
                    />
                </template>
                <span>Loading {{globalState.loading.join(", ")}}...</span>
            </v-tooltip>
            <router-view />
        </v-content>
        <!-- <v-footer app>Footer</v-footer> -->
    </v-app>
</template>

<script>
import globalState from "@/globalState.js";

globalState;

export default {
    name: "MorphoCluster",
    data: () => {
        return { globalState };
    },
    computed: {
        loading: function() {
            return globalState.loading;
        }
    }
};
</script>
<style>
.v-toolbar__title a {
    color: white;
}
</style>