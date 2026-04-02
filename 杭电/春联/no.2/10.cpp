#include <bits/stdc++.h>
using namespace std;
#define endl '\n'

const int N = 102;
const int mod = 1e9 + 7;
char mp[N][N];
int f[N][N][20][20][20]; // i j x y z; i行j列 x + y * z;

void solve(){
    memset(f, 0, sizeof(f));
    int n, m, k;
    cin >> n >> m >> k;
    for(int i = 1; i <= n; ++i)
        for(int j = 1 ; j <= m; ++j) cin >> mp[i][j];
        
    f[1][1][0][1][(mp[1][1]-'0')%k] = 1;

    auto cal = [&](int x, int y, int px, int py) {
        if(!isdigit(mp[x][y]) && !isdigit(mp[px][py])) return;
        if(isdigit(mp[x][y])){
            for(int i = 0; i < k; ++i){
                for(int j = 0; j < k; ++j){
                    for(int l = 0; l < k; ++l){
                        f[x][y][i][j][(l*10+(mp[x][y]-'0'))%k] += f[px][py][i][j][l];
                        f[x][y][i][j][(l*10+(mp[x][y]-'0'))%k] %= mod;
                    }
                }
            }
        }
        if(!isdigit(mp[x][y])){
            if(mp[x][y] == '+'){
                for(int i = 0; i < k; ++i){
                    for(int j = 0; j < k; ++j){
                        for(int l = 0; l < k; ++l){
                            f[x][y][(i+j*l)%k][1][0] += f[px][py][i][j][l];
                            f[x][y][(i+j*l)%k][1][0] %= mod;
                        }
                    }
                }
            }
            else if(mp[x][y] == '-'){
                for(int i = 0; i < k; ++i){
                    for(int j = 0; j < k; ++j){
                        for(int l = 0; l < k; ++l){
                            f[x][y][(i+j*l)%k][k-1][0] += f[px][py][i][j][l];
                            f[x][y][(i+j*l)%k][k-1][0] %= mod;
                        }
                    }
                }
            }
            else{
                for(int i = 0; i < k; ++i){
                    for(int j = 0; j < k; ++j){
                        for(int l = 0; l < k; ++l){
                            f[x][y][i][j*l%k][0] += f[px][py][i][j][l];
                            f[x][y][i][j*l%k][0] %= mod;
                        }
                    }
                }
            }
        }
    };

    for(int i = 1 ;i <= n ; ++i){
        for(int j = 1 ; j <= m; ++j){
            if(i!=1)
            cal(i,j,i-1,j);
            if(j!=1)
            cal(i,j,i,j-1);
        }
    }
    
    int ans = 0;
    for(int x = 0; x < k; ++x){
        for(int y = 0; y < k; ++y){
            for(int z = 0; z < k; ++z){
                if((x+y*z) % k == 0){
                    ans += f[n][m][x][y][z];
                    ans %= mod;
                }
            }
        }
    }
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    int T = 1;
    cin >> T;
    while(T--) solve();
    return 0;
}
    