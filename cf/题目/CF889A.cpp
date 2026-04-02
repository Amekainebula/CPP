#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
using namespace std;
void solve()
{
    int n;
    cin >> n;
    vector<int> a(n + 1);
    vector<pii>ans;
    int maxx = -INF;
    int now1=0;
    int minn = INF;
    int now2 = 0;
    bool flag = false;
    int cnt=0;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        if (a[i] > maxx)
        {
            maxx = a[i];
            now1 = i;
        }
        if (a[i] < minn)
        {
            minn = a[i];
            now2 = i;
        }
        if (i != 1 && !flag)
            if (a[i - 1] > a[i])
                flag = true;
    }
    if (flag)
    {
        for(int i=2;i<=n;i++)
        {
            while(a[i]<a[i-1])
            {
                cnt++;
                a[i]+=maxx;
                ans.pb({i,now2});
            }
        }
    }
    cout<<cnt<<endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}