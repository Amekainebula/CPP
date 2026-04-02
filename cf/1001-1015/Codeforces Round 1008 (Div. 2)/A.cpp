#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x,y,z) for(int(x)=(y);(x)<=(z);++(x))
#define ffg(x,y,z) for(int(x)=(y);(x)>=(z);--(x))
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
#define inf 0x7fffffff
#define endl '\n' 
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    int n,k;
    cin>>n>>k;
    vector<int> a(n+1);
    int sum=0;
    ff(i,1,n)
    {
        cin>>a[i];
        sum+=a[i];
    }
    
    cout<<((sum/n==k&&sum%n==0)?"YES":"NO")<<endl;
    
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