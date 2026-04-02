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
int n, m;
int isprime(int x)
{
    for (int i = 2; i * i <= x; i++)
    {
        if (x % i == 0)   
        {
            if(n%i==0&&m%i==0)return i;
        }
        if(i>n||i>m)return x;
    }
    return x;
}
void solve()
{
    
    cin >> n >> m;
    if (n == 1 || m == 1)
    {
        cout << "-1" << endl;
        return;
    }
    int t=n*m;
    int ans=isprime(t);
    cout << ans << endl;
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