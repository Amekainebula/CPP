#include <bits/stdc++.h>
#define int long long
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
#define yes cout << 'YES' << endl
#define no cout << 'NO' << endl
#define yess cout << 'Yes' << endl
#define noo cout << 'No' << endl
using namespace std;
vector<int> ans[100005];
void solve()
{
    int n, k;
    cin >> n >> k;
    for(int i=1;i<=n;i++)
    {
        cout<<(ans[n][i]==1?k:0)<<endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    for (int i = 1; i <= 100000; i++)
    {
        ans[i].pb(0);
        ans[i].pb(1);
        for (int j = 2; j <= i; j++)
        {
            if (ans[i - 1][j] == 0 && ans[i - 1][j - 1] == 0 
                || ans[i - 1][j] == 1 && ans[i - 1][j - 1] == 1)
                ans[i].pb(0);
            else
                ans[i].pb(1);
        }
        ans[i].pb(1);
    }
    while (T--)
    {
        solve();
    }
    return 0;
}